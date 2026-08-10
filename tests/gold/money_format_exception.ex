defmodule JodaMoney.Format.MoneyFormatException do
  @moduledoc """
  Raised when parsing or formatting money fails.

  Mirrors `org.joda.money.format.MoneyFormatException`.
  """

  defexception message: "Money formatting error"

  @impl true
  def exception(msg) when is_binary(msg) do
    %__MODULE__{message: msg}
  end

  def exception(opts) when is_list(opts) do
    %__MODULE__{message: Keyword.get(opts, :message, "Money formatting error")}
  end
end
