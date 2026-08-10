defmodule JodaMoney.CurrencyMismatchExceptionTest do
  use ExUnit.Case, async: true

  alias JodaMoney.CurrencyMismatchException

  describe "new/2" do
    test "with two currency codes builds a struct with both codes and a formatted message" do
      e = CurrencyMismatchException.new("USD", "EUR")
      assert e.first_currency == "USD"
      assert e.second_currency == "EUR"
      assert e.message == "Currencies differ: USD/EUR"
    end

    test "with nil first currency renders 'null' in the message but preserves nil in the struct" do
      e = CurrencyMismatchException.new(nil, "EUR")
      assert e.first_currency == nil
      assert e.second_currency == "EUR"
      assert e.message == "Currencies differ: null/EUR"
    end

    test "with nil second currency" do
      e = CurrencyMismatchException.new("USD", nil)
      assert e.first_currency == "USD"
      assert e.second_currency == nil
      assert e.message == "Currencies differ: USD/null"
    end

    test "with both currencies nil" do
      e = CurrencyMismatchException.new(nil, nil)
      assert e.first_currency == nil
      assert e.second_currency == nil
      assert e.message == "Currencies differ: null/null"
    end
  end

  describe "as an exception" do
    test "can be raised and caught with the correct message" do
      e = CurrencyMismatchException.new("USD", "JPY")

      caught =
        assert_raise CurrencyMismatchException, "Currencies differ: USD/JPY", fn ->
          raise e
        end

      assert caught.first_currency == "USD"
      assert caught.second_currency == "JPY"
    end

    test "Exception protocol returns the message" do
      e = CurrencyMismatchException.new("GBP", "CHF")
      assert Exception.message(e) == "Currencies differ: GBP/CHF"
    end
  end
end
