libarow-d
=========

D implementation of AROW linear classification.

## Install

Please copy src/arow.d onto your project directory.

## Example

```d
import arow;
import std.stdio;
import std.random;
import std.string;
import std.conv;
import std.file;
import std.stream;

struct example {
  int label;
  double[int] features;

  @safe nothrow {
    this(int label, double[int] features) {
      this.label = label;
      this.features = features;
    }
  }
}

double[int] parseLine(string line) {
  immutable string delim_value = ":";
  immutable string delim_cols = " ";

  string[] columns = line.split(delim_cols);

  double[int] f;

  for(int i = 1; i < columns.length; i++) {
    string[] arr = columns[i].split(delim_value);

    if(arr.length != 2)
      continue;

    assert(arr != null);
    assert(arr.length == 2);

    auto key = to!int(arr[0]);
    auto val = to!double(arr[1]);

    f[key] = val;
  }

  return f;
}


example[] readData(string filename){
  Stream file = new BufferedFile(filename);
  size_t num_lines = 0;

  example[] data;

  foreach (char[] _line; file) {

    string line = cast(string)_line;

    if (line.length == 0) continue;
    if (line[0] == '#') continue;

    assert(line[0] == '-' || line[0] == '+');

    int label = line[0] == '+' ? +1 : -1;
    double[int] vec = parseLine(line);

    if(vec != null) {
      example ex = example(label, vec);

      assert(vec != null);
      assert(ex.features != null);
      assert(ex.label == -1 || ex.label == +1);

      data ~= ex;
    
    }
  }

  file.close();

  return data;
}

void main(string[] args) {
  immutable uint dimension = 1355192;
  Arow arow = new Arow(dimension, 0.1);
  example[] data = readData("news20.binary");

  data.randomShuffle(Random());

  int i;
  int num_train = 15000;
  for(i = 0; i < num_train; i++)
    arow.update(data[i].features, data[i].label);
  
  int correct = 0;
  for(; i < data.length; i++) {
    auto result = arow.predict(data[i].features);
    if(result == data[i].label) {
      correct++;
    }
  }

  writeln(correct, "/", (cast(double)data.length - num_train));
  writeln(correct / (cast(double)data.length - num_train));
}
```

## Licenes
libarow-d is released under the [MIT License](http://www.opensource.org/licenses/MIT).
