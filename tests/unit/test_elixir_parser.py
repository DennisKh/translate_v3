"""language/elixir.py — module shape extraction."""

from language.elixir import FunctionSig, describe_source


def test_simple_module():
    src = '''
defmodule App.Foo do
  def hello(name) do
    "hi #{name}"
  end
end
'''
    shape = describe_source(src)
    assert shape.module_name == "App.Foo"
    assert shape.kind == "module"
    assert FunctionSig("hello", 1) in shape.public_functions


def test_nested_module_name():
    src = "defmodule A.B.C do\n  def x(), do: :ok\nend\n"
    shape = describe_source(src)
    assert shape.module_name == "A.B.C"


def test_paren_and_no_paren_defs():
    src = """
defmodule M do
  def a(), do: 1
  def b do
    2
  end
  def c(x), do: x
end
"""
    shape = describe_source(src)
    names = {(s.name, s.arity) for s in shape.public_functions}
    assert ("a", 0) in names
    assert ("b", 0) in names
    assert ("c", 1) in names


def test_default_args_produce_multiple_arities():
    src = """
defmodule M do
  def greet(name, greeting \\\\ "hello"), do: greeting <> " " <> name
end
"""
    shape = describe_source(src)
    names = {(s.name, s.arity) for s in shape.public_functions}
    assert ("greet", 1) in names
    assert ("greet", 2) in names


def test_defstruct_fields_list_form():
    src = """
defmodule M do
  defstruct [:a, :b, :c]
end
"""
    shape = describe_source(src)
    assert shape.defstruct_fields == ["a", "b", "c"]


def test_defstruct_fields_keyword_form():
    src = """
defmodule M do
  defstruct name: "", value: nil
end
"""
    shape = describe_source(src)
    assert set(shape.defstruct_fields) == {"name", "value"}


def test_protocol_detected():
    src = """
defprotocol P do
  def do_it(x)
end
"""
    shape = describe_source(src)
    assert shape.kind == "protocol"


def test_behaviour_detected_via_callback():
    src = """
defmodule B do
  @callback do_it(x :: any()) :: :ok
end
"""
    shape = describe_source(src)
    assert shape.kind == "behaviour"
    assert FunctionSig("do_it", 1) in shape.callbacks


def test_non_module_returns_none():
    src = "just some text, no defmodule\n"
    assert describe_source(src) is None


def test_line_count():
    src = "defmodule M do\n  def x, do: 1\nend\n"
    shape = describe_source(src)
    assert shape.line_count == 4


def test_bang_and_question_names():
    src = """
defmodule M do
  def valid?(x), do: is_binary(x)
  def parse!(s), do: s
end
"""
    shape = describe_source(src)
    names = {(s.name, s.arity) for s in shape.public_functions}
    assert ("valid?", 1) in names
    assert ("parse!", 1) in names
