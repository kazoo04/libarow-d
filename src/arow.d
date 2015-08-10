/**
 * A Library for AROW Linear Classification 
 *
 * Authors: Kazuya Gokita
 */

module arow;

import std.math;
import std.random;

/***
 * Adaptive Regularization of Weight Vectors
 *
 * See_Also:
 *   K. Crammer, A. Kulesza, and M. Dredze. "Adaptive regularization of weight vectors" NIPS 2009
 */
class Arow {
  private:
    size_t dimension; // Size of feature vector
    double[] mean;    // Average vector
    double[] cov;     // Variance matrix (diagonal)

    double r;         // Hyper parameter ( r > 0 )

    invariant() {
      assert(mean != null);
      assert(cov != null);
      assert(mean.length == dimension);
      assert(cov.length == dimension);
      assert(r > 0);
    }


    /**
     * Calculate the distance between a vector and the hyperplane
     * Params:
     *  f = feature
     *
     * Returns: Margin(Euclidean distance)
     */
    double getMargin(in double[int] f) @trusted
    in {
      assert(f != null);
    }
    out(margin) {
      assert(margin != double.nan);
      assert(margin != double.nan && margin != -double.infinity);
    }
    body {
      double margin = 0.0;
      foreach(index; f.keys) {
        margin += mean[index] * f[index];
      }

      return margin;
    }


    /**
     * Calculate confidence
     * Params:
     *  f = feature
     *
     * Returns: confidence
     */ 
    double getConfidence(in double[int] f) @trusted
    in {
      assert(f != null);
    }
    out(confidence) {
      assert(confidence != double.nan);
      assert(confidence != double.nan && confidence != -double.infinity);
    }
    body {
      double confidence = 0.0;
      foreach(index; f.keys) {
        confidence += cov[index] * f[index] * f[index];
      }

      return confidence;
    }


  public:
    this(size_t num_features, double param = 0.1) {
      dimension = num_features;
      mean = new double[dimension];
      cov = new double[dimension];

      mean[] = 0.0;
      cov[] = 1.0;
      r = param;
    }

    
    this(size_t size, double[] mean, double[] cov, double hyperparameter) {
      this.dimension = size;
      this.mean      = mean;
      this.cov       = cov;
      this.r         = hyperparameter;
    }


    @property {
      auto dim() { return dimension; }
    }


    @property {
      auto param() { return r; }
      auto param(double val) { return r = val; }
    }


    /**
     * Update weight vector
     * Params:
     *  f     = feature
     *  label = class label (+1 / -1)
     *
     * Returns: loss (0 / 1)
     */
    int update(in double[int] f, int label) @trusted
    in {
      assert(label == -1 || label == 1);
      assert(f != null);
    }
    out(loss) {
      assert(loss == 0 || loss == 1);
    }
    body {
      immutable margin = getMargin(f);
      if (margin * label >= 1) return 0;

      immutable confidence = getConfidence(f);
      immutable beta = 1.0 / (confidence + r);
      immutable alpha = (1.0 - label * margin) * beta;

      // Update mean
      foreach(index; f.keys) {
        mean[index] += alpha * cov[index] * label * f[index];
      }

      // Update covariance
      foreach(index; f.keys) {
        cov[index] = 1.0 / ((1.0 / cov[index]) + f[index] * f[index] / r);
      }

      return margin * label < 0 ? 1 : 0;
    }


    /**
     * Predict
     * Params:
     *  f = feature vector
     * Returns: class label (-1, 1)
     */
    int predict(in double[int] f) @trusted
    in {
      assert(f != null);
    }
    out(label) {
      assert(label == -1 || label == 1);
    }
    body {
      double m = getMargin(f);
      return m > 0 ? 1 : -1;
    }


    /**
     * Write AROW data to the specified file
     * Params:
     *  filename = output path
     */
    void saveArow(in immutable string filename)
    in {
      assert(filename != null);
    }
    body
    {
      import std.stream;
      auto buffer = new BufferedFile();
      buffer.create(filename);
      buffer.write(dimension);
      buffer.write(r);
      foreach (m; mean) buffer.write(m);
      foreach (c; cov)  buffer.write(c);
      buffer.close();
    }


    /**
     * Load AROW data from the specified file
     * Params:
     *  filename = output path
     *
     * Returns: Arow instance
     */
    Arow loadArowFile(in string filename)
    {
      import std.stream;
      Stream file = new BufferedFile(filename);
      scope(exit) file.close();

      size_t size;
      file.read(size);

      double hyperparameter;
      file.read(hyperparameter);

      double[] mean = new double[size];
      for (size_t i = 0; i < size; i++) file.read(mean[i]);

      double[] cov = new double[size];
      for (size_t i = 0; i < size; i++) file.read(cov[i]);

      return new Arow(size, mean, cov, hyperparameter);
    }


    Arow opBinary(string op)(Arow b) {
      static if(op != "+") static assert(0, "Operator " ~ op ~ " not implemented");

      if(this.dimension != b.dimension) {
        throw new Exception("Vectors must be the same length");
      }

      auto arow = new Arow(dimension, r);

      for (size_t i = 0; i < dimension; i++) {
        arow.mean[i] = (mean[i] + b.mean[i]) / 2;
        arow.cov[i] = (cov[i] + b.cov[i]) / 2;
      }

      return arow;
    }

    unittest {
      double[] a_m = [1, 2, 3], a_c = [4, 5, 6];
      double[] b_m = [7, 8, 9], b_c = [10, 11, 12];

      Arow a = new Arow(3, a_m, a_c, 1);
      Arow b = new Arow(3, b_m, b_c, 1);

      Arow c = a + b;
      assert(c.mean[0] == 4);
      assert(c.mean[1] == 5);
      assert(c.mean[2] == 6);
    }


    Arow opOpAssign(string op)(Arow b) {
      static if(op != "+") static assert(0, "Operator " ~ op ~ " not implemented");

      if(this.dimension != b.dimension) {
        throw new Exception("Vectors must be the same length");
      }

      for (size_t i = 0; i < dimension; i++) {
        mean[i] = (mean[i] + b.mean[i]) / 2;
        cov[i] = (cov[i] + b.cov[i]) / 2;
      }

      return this;
    }

    unittest {
      double[] a_m = [1, 2, 3], a_c = [4, 5, 6];
      double[] b_m = [7, 8, 9], b_c = [10, 11, 12];

      Arow a = new Arow(3, a_m, a_c, 1);
      Arow b = new Arow(3, b_m, b_c, 1);

      a += b;
      assert(a.mean[0] == 4);
      assert(a.mean[1] == 5);
      assert(a.mean[2] == 6);
    }
}

