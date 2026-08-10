defmodule JodaMoney.CurrencyMismatchException do
  @moduledoc """
  Exception raised when a monetary operation fails due to mismatched currencies.

  For example, this exception would be raised when trying to add a monetary
  value in one currency to a monetary value in a different currency.
  """

  defexception [:message, :first_currency, :second_currency]

  @doc """
  Creates a new `CurrencyMismatchException` with the two mismatched currency codes.

  `first_currency` and `second_currency` should be currency code strings (or nil).
  """
  def new(first_currency, second_currency) do
    first_code = first_currency || "null"
    second_code = second_currency || "null"
    msg = "Currencies differ: #{first_code}/#{second_code}"

    %__MODULE__{
      message: msg,
      first_currency: first_currency,
      second_currency: second_currency
    }
  end
end
