#pragma once

#include <vector>

struct Point { 
  int t;
  double x, y;

  Point(int t, double x, double y): t(t), x(x), y(y) {}
  Point(): Point(0, 0., 0.) {}
};

struct PolyLine { 
  double x, y, theta, kappa;

  PolyLine(double x, double y, double theta, double kappa): x(x), y(y), theta(theta), kappa(kappa) {}
  PolyLine(): PolyLine(0., 0., 0., 0.) {}
};

int t0 = 0;
std::vector<Point> pts;
std::vector<Point> fit;
std::vector<PolyLine> sp; // max t up to the beginning of the i-th segment