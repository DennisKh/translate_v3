defmodule JodaMoney.Format.GroupingStyle do
  @moduledoc """
  Grouping styles for numeric formatting.

  Mirrors `org.joda.money.format.GroupingStyle`:
    - `:none`   — no grouping
    - `:full`   — group every N digits (e.g., 1,000,000)
    - `:before_decimal_point` — group only before the decimal
  """

  @type t :: :none | :full | :before_decimal_point

  @all [:none, :full, :before_decimal_point]

  @spec values() :: [t()]
  def values, do: @all

  @spec valid?(any()) :: boolean()
  def valid?(style) when style in @all, do: true
  def valid?(_), do: false
end
