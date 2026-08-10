defmodule JodaMoney.Format.MoneyFormatExceptionTest do
  use ExUnit.Case, async: true

  alias JodaMoney.Format.MoneyFormatException

  test "raise/1 with a string message" do
    err =
      try do
        raise MoneyFormatException, "bad input"
      rescue
        e -> e
      end

    assert err.message == "bad input"
  end

  test "raise/1 with keyword opts" do
    err =
      try do
        raise MoneyFormatException, message: "custom"
      rescue
        e -> e
      end

    assert err.message == "custom"
  end

  test "Exception.message/1 works" do
    err = %MoneyFormatException{message: "hi"}
    assert Exception.message(err) == "hi"
  end
end
