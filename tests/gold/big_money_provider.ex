defprotocol JodaMoney.BigMoneyProvider do
  @moduledoc """
  Protocol providing a uniform interface to obtain a `BigMoney`.

  Both `JodaMoney.Money` and `JodaMoney.BigMoney` implement this protocol.
  """

  @doc """
  Returns a `BigMoney` equivalent to the value of this structure.
  """
  def to_big_money(provider)
end
