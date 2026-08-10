defmodule JodaMoney.CurrencyUnit do
  @moduledoc """
  A unit of currency.

  This module represents a unit of currency such as the British Pound, Euro
  or US Dollar.

  Currency data is loaded at compile time from `priv/CurrencyData.csv` and
  `priv/CountryData.csv`. Runtime registration is supported via `register_currency/4`.
  """

  alias JodaMoney.IllegalCurrencyException

  @enforce_keys [:code, :numeric_code, :decimal_places]
  defstruct [:code, :numeric_code, :decimal_places]

  @type t :: %__MODULE__{
          code: String.t(),
          numeric_code: integer(),
          decimal_places: integer()
        }

  # ── Compile-time currency data ─────────────────────────────────────────

  @external_resource "priv/CurrencyData.csv"
  @external_resource "priv/CountryData.csv"

  # Map of currency_code => {numeric_code, decimal_places}
  @static_by_code "priv/CurrencyData.csv"
                  |> File.read!()
                  |> String.split(~r/\R/)
                  |> Enum.reduce(%{}, fn line, acc ->
                    case Regex.run(~r/^([A-Z]{3}),(-1|[0-9]{1,3}),(-1|[0-9]|[1-2][0-9]|30)/, line) do
                      [_, code, num, dec] ->
                        Map.put(acc, code, {String.to_integer(num), String.to_integer(dec)})

                      _ ->
                        acc
                    end
                  end)

  # Map of numeric_code => currency_code
  @static_by_numeric Enum.reduce(@static_by_code, %{}, fn {code, {numeric, _dec}}, acc ->
                       if numeric >= 0, do: Map.put(acc, numeric, code), else: acc
                     end)

  # Map of country_code => currency_code
  @static_by_country "priv/CountryData.csv"
                     |> File.read!()
                     |> String.split(~r/\R/)
                     |> Enum.reduce(%{}, fn line, acc ->
                       case Regex.run(~r/^([A-Z]{2}),([A-Z]{3})/, line) do
                         [_, country, currency] -> Map.put(acc, country, currency)
                         _ -> acc
                       end
                     end)

  # ── ETS table names for runtime registration ────────────────────────────

  @ets_by_code :joda_money_cu_by_code
  @ets_by_numeric :joda_money_cu_by_numeric
  @ets_by_country :joda_money_cu_by_country

  # ── Validation regex ────────────────────────────────────────────────────

  @code_pattern ~r/\A[A-Z][A-Z][A-Z]\z/

  # ── Registered currencies / countries ──────────────────────────────────

  @doc """
  Gets the list of all registered currencies, sorted alphabetically by code.
  """
  @spec registered_currencies() :: [t()]
  def registered_currencies do
    static = Enum.map(@static_by_code, fn {code, {numeric, dec}} -> build(code, numeric, dec) end)

    runtime =
      case :ets.whereis(@ets_by_code) do
        :undefined ->
          []

        _ ->
          :ets.tab2list(@ets_by_code)
          |> Enum.map(fn {code, numeric, dec} -> build(code, numeric, dec) end)
      end

    (static ++ runtime)
    |> Enum.uniq_by(& &1.code)
    |> Enum.sort_by(& &1.code)
  end

  @doc """
  Gets the list of all registered country codes, sorted alphabetically.
  """
  @spec registered_countries() :: [String.t()]
  def registered_countries do
    static_keys = Map.keys(@static_by_country)

    runtime_keys =
      case :ets.whereis(@ets_by_country) do
        :undefined -> []
        _ -> Enum.map(:ets.tab2list(@ets_by_country), fn {country, _} -> country end)
      end

    (static_keys ++ runtime_keys)
    |> Enum.uniq()
    |> Enum.sort()
  end

  # ── Registration ────────────────────────────────────────────────────────

  @doc """
  Registers a currency allowing it to be used, with optional country codes.

  Set `force: true` to replace any existing matching currency (default: false).
  """
  @spec register_currency(String.t(), integer(), integer(), [String.t()], boolean()) :: t()
  def register_currency(
        currency_code,
        numeric_currency_code,
        decimal_places,
        country_codes \\ [],
        force \\ false
      ) do
    validate_register_args!(currency_code, numeric_currency_code, decimal_places, country_codes)
    ensure_runtime_tables()
    unless force, do: check_not_duplicate!(currency_code, numeric_currency_code, country_codes)
    persist_registration(currency_code, numeric_currency_code, decimal_places, country_codes)
    of(currency_code)
  end

  defp validate_register_args!(
         currency_code,
         numeric_currency_code,
         decimal_places,
         country_codes
       ) do
    if is_nil(currency_code), do: raise(ArgumentError, "Currency code must not be null")

    if String.length(currency_code) != 3,
      do: raise(ArgumentError, "Invalid string code, must be length 3")

    unless Regex.match?(@code_pattern, currency_code),
      do: raise(ArgumentError, "Invalid string code, must be ASCII upper-case letters")

    if numeric_currency_code < -1 or numeric_currency_code > 999,
      do: raise(ArgumentError, "Invalid numeric code")

    if decimal_places < -1 or decimal_places > 30,
      do: raise(ArgumentError, "Invalid number of decimal places")

    if is_nil(country_codes), do: raise(ArgumentError, "Country codes must not be null")
  end

  defp check_not_duplicate!(currency_code, numeric_currency_code, country_codes) do
    if Map.has_key?(@static_by_code, currency_code) or ets_has_code?(currency_code),
      do: raise(ArgumentError, "Currency already registered: #{currency_code}")

    if numeric_currency_code >= 0 and
         (Map.has_key?(@static_by_numeric, numeric_currency_code) or
            ets_has_numeric?(numeric_currency_code)),
       do: raise(ArgumentError, "Currency already registered: #{currency_code}")

    Enum.each(country_codes, fn cc ->
      if Map.has_key?(@static_by_country, cc) or ets_has_country?(cc),
        do: raise(ArgumentError, "Currency already registered for country: #{cc}")
    end)
  end

  defp persist_registration(currency_code, numeric_currency_code, decimal_places, country_codes) do
    :ets.insert(@ets_by_code, {currency_code, numeric_currency_code, decimal_places})

    if numeric_currency_code >= 0 do
      :ets.insert(@ets_by_numeric, {numeric_currency_code, currency_code})
    end

    Enum.each(country_codes, fn cc ->
      :ets.insert(@ets_by_country, {cc, currency_code})
    end)
  end

  @doc """
  Registers a country code and its associated currency.
  """
  @spec register_country(String.t(), t()) :: :ok
  def register_country(country_code, %__MODULE__{code: currency_code}) do
    ensure_runtime_tables()
    :ets.insert(@ets_by_country, {country_code, currency_code})
    :ok
  end

  # ── Lookup / factory ────────────────────────────────────────────────────

  @doc """
  Obtains a `CurrencyUnit` for the given three-letter ISO-4217 code.

  Raises `IllegalCurrencyException` if the code is unknown.
  """
  @spec of(String.t()) :: t()
  def of(currency_code) when is_binary(currency_code) do
    if is_nil(currency_code), do: raise(ArgumentError, "Currency code must not be null")

    case Map.get(@static_by_code, currency_code) do
      {numeric, dec} ->
        build(currency_code, numeric, dec)

      nil ->
        case runtime_lookup_by_code(currency_code) do
          nil ->
            raise IllegalCurrencyException, message: "Unknown currency '#{currency_code}'"

          unit ->
            unit
        end
    end
  end

  def of(nil), do: raise(ArgumentError, "Currency code must not be null")

  @doc """
  Obtains a `CurrencyUnit` for the given ISO-4217 numeric code (as string or integer).
  """
  @spec of_numeric_code(String.t() | integer()) :: t()
  def of_numeric_code(code) when is_binary(code) do
    if is_nil(code), do: raise(ArgumentError, "Currency code must not be null")

    int_code =
      case String.length(code) do
        1 -> String.to_integer(code)
        2 -> String.to_integer(code)
        3 -> String.to_integer(code)
        _ -> raise IllegalCurrencyException, message: "Unknown currency '#{code}'"
      end

    of_numeric_code(int_code)
  end

  def of_numeric_code(nil), do: raise(ArgumentError, "Currency code must not be null")

  def of_numeric_code(numeric_code) when is_integer(numeric_code) do
    case Map.get(@static_by_numeric, numeric_code) do
      nil ->
        case runtime_lookup_by_numeric(numeric_code) do
          nil ->
            raise IllegalCurrencyException,
              message: "Unknown currency '#{numeric_code}'"

          unit ->
            unit
        end

      code ->
        of(code)
    end
  end

  @doc """
  Obtains a `CurrencyUnit` for the given ISO-3166 country code.

  Raises `IllegalCurrencyException` if no currency is registered for the country.
  """
  @spec of_country(String.t()) :: t()
  def of_country(country_code) when is_binary(country_code) do
    if is_nil(country_code), do: raise(ArgumentError, "Country code must not be null")

    case Map.get(@static_by_country, country_code) do
      nil ->
        case runtime_lookup_by_country(country_code) do
          nil ->
            raise IllegalCurrencyException,
              message: "No currency found for country '#{country_code}'"

          unit ->
            unit
        end

      currency_code ->
        of(currency_code)
    end
  end

  def of_country(nil), do: raise(ArgumentError, "Country code must not be null")

  # ── Instance functions ──────────────────────────────────────────────────

  @doc "Gets the ISO-4217 three-letter currency code."
  @spec code(t()) :: String.t()
  def code(%__MODULE__{code: c}), do: c

  @doc "Gets the ISO-4217 numeric currency code. Returns -1 if no numeric code."
  @spec numeric_code(t()) :: integer()
  def numeric_code(%__MODULE__{numeric_code: n}), do: n

  @doc """
  Gets the ISO-4217 numeric currency code as a zero-padded three-digit string.
  Returns an empty string if there is no numeric code.
  """
  @spec numeric3_code(t()) :: String.t()
  def numeric3_code(%__MODULE__{numeric_code: n}) when n < 0, do: ""

  def numeric3_code(%__MODULE__{numeric_code: n}) do
    n |> Integer.to_string() |> String.pad_leading(3, "0")
  end

  @doc """
  Gets the number of decimal places typically used by this currency.
  Pseudo-currencies return 0.
  """
  @spec decimal_places(t()) :: integer()
  def decimal_places(%__MODULE__{decimal_places: d}) when d < 0, do: 0
  def decimal_places(%__MODULE__{decimal_places: d}), do: d

  @doc "Returns true if this is a pseudo-currency (negative decimal_places)."
  @spec pseudo_currency?(t()) :: boolean()
  def pseudo_currency?(%__MODULE__{decimal_places: d}), do: d < 0

  @doc """
  Gets the set of country codes registered for this currency.
  """
  @spec country_codes(t()) :: MapSet.t(String.t())
  def country_codes(%__MODULE__{code: currency_code}) do
    static_codes =
      @static_by_country
      |> Enum.filter(fn {_cc, cur} -> cur == currency_code end)
      |> Enum.map(fn {cc, _} -> cc end)

    runtime_codes =
      case :ets.whereis(@ets_by_country) do
        :undefined ->
          []

        _ ->
          :ets.tab2list(@ets_by_country)
          |> Enum.filter(fn {_cc, cur} -> cur == currency_code end)
          |> Enum.map(fn {cc, _} -> cc end)
      end

    MapSet.new(static_codes ++ runtime_codes)
  end

  @doc """
  Gets the currency symbol. Falls back to the currency code if no symbol is available.
  The optional `locale` argument is accepted for API compatibility but ignored.
  """
  @spec symbol(t(), any()) :: String.t()
  def symbol(%__MODULE__{code: code}, _locale \\ nil) do
    # Elixir has no JDK Currency equivalent; common symbols are provided inline.
    currency_symbol(code)
  end

  defp currency_symbol("USD"), do: "$"
  defp currency_symbol("EUR"), do: "\u20AC"
  defp currency_symbol("GBP"), do: "\u00A3"
  defp currency_symbol("JPY"), do: "\u00A5"
  defp currency_symbol("CHF"), do: "CHF"
  defp currency_symbol("AUD"), do: "A$"
  defp currency_symbol("CAD"), do: "CA$"
  defp currency_symbol("CNY"), do: "\u00A5"
  defp currency_symbol("INR"), do: "\u20B9"
  defp currency_symbol("XXX"), do: "XXX"
  defp currency_symbol(other), do: other

  @doc "Compares two currencies alphabetically by code."
  @spec compare(t(), t()) :: :lt | :eq | :gt
  def compare(%__MODULE__{code: a}, %__MODULE__{code: b}) do
    cond do
      a < b -> :lt
      a > b -> :gt
      true -> :eq
    end
  end

  # ── String.Chars ────────────────────────────────────────────────────────

  defimpl String.Chars do
    def to_string(%JodaMoney.CurrencyUnit{code: code}), do: code
  end

  # ── Inspect ─────────────────────────────────────────────────────────────

  defimpl Inspect do
    def inspect(%JodaMoney.CurrencyUnit{code: code}, _opts), do: "#CurrencyUnit<#{code}>"
  end

  # ── Private helpers ─────────────────────────────────────────────────────

  defp build(code, numeric, dec),
    do: %__MODULE__{code: code, numeric_code: numeric, decimal_places: dec}

  defp ensure_runtime_tables do
    for table <- [@ets_by_code, @ets_by_numeric, @ets_by_country] do
      case :ets.whereis(table) do
        :undefined ->
          try do
            :ets.new(table, [:set, :public, :named_table, {:write_concurrency, true}])
          rescue
            _ -> :ok
          end

        _ ->
          :ok
      end
    end
  end

  defp ets_has_code?(code) do
    case :ets.whereis(@ets_by_code) do
      :undefined -> false
      _ -> :ets.member(@ets_by_code, code)
    end
  end

  defp ets_has_numeric?(numeric) do
    case :ets.whereis(@ets_by_numeric) do
      :undefined -> false
      _ -> :ets.member(@ets_by_numeric, numeric)
    end
  end

  defp ets_has_country?(country) do
    case :ets.whereis(@ets_by_country) do
      :undefined -> false
      _ -> :ets.member(@ets_by_country, country)
    end
  end

  defp runtime_lookup_by_code(code) do
    case :ets.whereis(@ets_by_code) do
      :undefined ->
        nil

      _ ->
        case :ets.lookup(@ets_by_code, code) do
          [{^code, numeric, dec}] -> build(code, numeric, dec)
          _ -> nil
        end
    end
  end

  defp runtime_lookup_by_numeric(numeric) do
    case :ets.whereis(@ets_by_numeric) do
      :undefined ->
        nil

      _ ->
        case :ets.lookup(@ets_by_numeric, numeric) do
          [{^numeric, code}] -> of(code)
          _ -> nil
        end
    end
  end

  defp runtime_lookup_by_country(country) do
    case :ets.whereis(@ets_by_country) do
      :undefined ->
        nil

      _ ->
        case :ets.lookup(@ets_by_country, country) do
          [{^country, code}] -> of(code)
          _ -> nil
        end
    end
  end
end
