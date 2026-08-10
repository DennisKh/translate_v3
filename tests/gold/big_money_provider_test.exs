defmodule JodaMoney.BigMoneyProviderTest do
  use ExUnit.Case, async: true

  alias JodaMoney.BigMoneyProvider

  # Protocol dispatch cannot be tested via inline `defimpl` in a test module
  # when the project already consolidated protocols (which is the default
  # in mix projects). We test the protocol shape via introspection instead —
  # this is the invariant a translator must produce.

  describe "protocol shape" do
    test "protocol module is defined via defprotocol" do
      assert function_exported?(BigMoneyProvider, :__protocol__, 1)
    end

    test "declares exactly the to_big_money/1 function" do
      assert BigMoneyProvider.__protocol__(:functions) == [to_big_money: 1]
    end

    test "is a protocol (not a behaviour or plain module)" do
      # __protocol__(:module) returns the protocol name; only defprotocol
      # emits this callback.
      assert BigMoneyProvider.__protocol__(:module) == BigMoneyProvider
    end
  end

  describe "protocol enforcement" do
    test "calling on a type with no impl raises Protocol.UndefinedError" do
      assert_raise Protocol.UndefinedError, fn ->
        BigMoneyProvider.to_big_money(42)
      end

      assert_raise Protocol.UndefinedError, fn ->
        BigMoneyProvider.to_big_money("plain string")
      end

      assert_raise Protocol.UndefinedError, fn ->
        BigMoneyProvider.to_big_money(%{fake: :map})
      end
    end
  end
end
