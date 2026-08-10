defmodule JodaMoney.IllegalCurrencyException do
  @moduledoc """
  Raised when an ISO-4217 currency code is not recognized.

  Mirrors `org.joda.money.IllegalCurrencyException` from the Java library.
  """

  defexception [:message, :code]

  @impl true
  def exception(opts) when is_list(opts) do
    code = Keyword.get(opts, :code)
    message = Keyword.get_lazy(opts, :message, fn -> default_message(code) end)
    %__MODULE__{message: message, code: code}
  end

  def exception(code) when is_binary(code) do
    %__MODULE__{message: default_message(code), code: code}
  end

  defp default_message(nil), do: "Unknown currency"
  defp default_message(code), do: "Unknown currency '#{code}'"
end
