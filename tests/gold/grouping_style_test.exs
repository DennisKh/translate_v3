defmodule JodaMoney.Format.GroupingStyleTest do
  use ExUnit.Case, async: true

  alias JodaMoney.Format.GroupingStyle

  test "values/0 lists all styles" do
    values = GroupingStyle.values()
    assert :none in values
    assert :full in values
    assert :before_decimal_point in values
  end

  describe "valid?/1" do
    test "accepts the three styles" do
      assert GroupingStyle.valid?(:none)
      assert GroupingStyle.valid?(:full)
      assert GroupingStyle.valid?(:before_decimal_point)
    end

    test "rejects unknown values" do
      refute GroupingStyle.valid?(:something_else)
      refute GroupingStyle.valid?("full")
      refute GroupingStyle.valid?(nil)
    end
  end
end
