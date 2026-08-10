defmodule JodaMoney.CurrencyUnitTest do
  use ExUnit.Case, async: false
  # async: false — this module uses shared ETS tables (registered_currencies).

  alias JodaMoney.CurrencyUnit

  describe "of/1" do
    test "returns a CurrencyUnit for a known currency code" do
      usd = CurrencyUnit.of("USD")
      assert %CurrencyUnit{code: "USD"} = usd
      assert CurrencyUnit.code(usd) == "USD"
    end

    test "returns a CurrencyUnit with correct numeric code for USD" do
      usd = CurrencyUnit.of("USD")
      assert CurrencyUnit.numeric_code(usd) == 840
    end

    test "returns a CurrencyUnit with 2 decimal places for USD" do
      usd = CurrencyUnit.of("USD")
      assert CurrencyUnit.decimal_places(usd) == 2
    end

    test "returns a CurrencyUnit for EUR" do
      eur = CurrencyUnit.of("EUR")
      assert CurrencyUnit.code(eur) == "EUR"
      assert CurrencyUnit.numeric_code(eur) == 978
    end

    test "returns a CurrencyUnit for JPY (0 decimal places)" do
      jpy = CurrencyUnit.of("JPY")
      assert CurrencyUnit.code(jpy) == "JPY"
      assert CurrencyUnit.decimal_places(jpy) == 0
    end

    test "raises for unknown currency code" do
      assert_raise JodaMoney.IllegalCurrencyException, fn ->
        CurrencyUnit.of("XXX_NOT_A_REAL_CODE")
      end
    end

    test "raises ArgumentError for nil input" do
      assert_raise ArgumentError, fn ->
        CurrencyUnit.of(nil)
      end
    end
  end

  describe "of_numeric_code/1" do
    test "returns the correct currency for USD numeric code (840)" do
      usd = CurrencyUnit.of_numeric_code(840)
      assert CurrencyUnit.code(usd) == "USD"
    end

    test "accepts a stringified numeric code" do
      usd = CurrencyUnit.of_numeric_code("840")
      assert CurrencyUnit.code(usd) == "USD"
    end
  end

  describe "of_country/1" do
    test "returns the currency for a known country" do
      currency = CurrencyUnit.of_country("US")
      assert CurrencyUnit.code(currency) == "USD"
    end
  end

  describe "field accessors" do
    setup do
      {:ok, usd: CurrencyUnit.of("USD"), jpy: CurrencyUnit.of("JPY")}
    end

    test "code/1 returns the ISO code", %{usd: usd} do
      assert CurrencyUnit.code(usd) == "USD"
    end

    test "numeric3_code/1 formats numeric code as 3-digit string", %{usd: usd} do
      assert CurrencyUnit.numeric3_code(usd) == "840"
    end

    test "decimal_places/1 returns non-negative decimals", %{jpy: jpy} do
      assert CurrencyUnit.decimal_places(jpy) == 0
    end

    test "pseudo_currency?/1 is false for real currencies", %{usd: usd, jpy: jpy} do
      refute CurrencyUnit.pseudo_currency?(usd)
      refute CurrencyUnit.pseudo_currency?(jpy)
    end
  end

  describe "compare/2" do
    test "returns :eq for equal codes" do
      usd1 = CurrencyUnit.of("USD")
      usd2 = CurrencyUnit.of("USD")
      assert CurrencyUnit.compare(usd1, usd2) == :eq
    end

    test "sorts alphabetically" do
      eur = CurrencyUnit.of("EUR")
      usd = CurrencyUnit.of("USD")
      assert CurrencyUnit.compare(eur, usd) == :lt
      assert CurrencyUnit.compare(usd, eur) == :gt
    end
  end

  describe "struct enforcement" do
    test "cannot build a CurrencyUnit without required fields" do
      assert_raise ArgumentError, fn ->
        struct!(CurrencyUnit, %{})
      end
    end

    test "requires code, numeric_code, decimal_places (@enforce_keys)" do
      assert_raise ArgumentError, fn ->
        struct!(CurrencyUnit, %{code: "AAA"})
      end
    end
  end
end
