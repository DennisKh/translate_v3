defmodule JodaMoney.IllegalCurrencyExceptionTest do
  use ExUnit.Case, async: true

  alias JodaMoney.IllegalCurrencyException

  describe "raise/1 with a code string" do
    test "sets the code and default message" do
      err =
        try do
          raise IllegalCurrencyException, "XYZ"
        rescue
          e -> e
        end

      assert err.code == "XYZ"
      assert err.message =~ "XYZ"
    end
  end

  describe "raise/1 with keyword opts" do
    test "custom message overrides default" do
      err =
        try do
          raise IllegalCurrencyException, code: "ZZZ", message: "custom text"
        rescue
          e -> e
        end

      assert err.code == "ZZZ"
      assert err.message == "custom text"
    end

    test "code alone falls back to default message" do
      err =
        try do
          raise IllegalCurrencyException, code: "ABC"
        rescue
          e -> e
        end

      assert err.code == "ABC"
      assert err.message =~ "ABC"
    end
  end

  describe "Exception protocol" do
    test "Exception.message/1 returns the message" do
      err = %IllegalCurrencyException{message: "hello", code: "USD"}
      assert Exception.message(err) == "hello"
    end
  end
end
