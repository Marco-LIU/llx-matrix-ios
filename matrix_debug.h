//
// Created by Marco.LIU on 16/8/30.
// Copyright (c) 2016 LLX. All rights reserved.
//

#ifndef _LLX_UTIL_MATRIX_MATRIX_DEBUG_H_
#define _LLX_UTIL_MATRIX_MATRIX_DEBUG_H_

#include "matrices.h"

#include <iostream>
#include <string>

inline std::ostream& operator<<(std::ostream& out, const LLX::util::Vector& v) {
  return out << "(" << LLX::util::VectorGetX(v)
             << ", " << LLX::util::VectorGetY(v)
             << ", " << LLX::util::VectorGetZ(v)
             << ", " << LLX::util::VectorGetW(v) << ")";
}

inline std::ostream& operator<<(std::ostream& out, const LLX::util::Matrix& m) {
  out << "\n";
  char buf[128];
  sprintf(buf, "\t%10.6f %10.6f %10.6f %10.6f\n",
          LLX::util::VectorGetX(m.c[0]), LLX::util::VectorGetX(m.c[1]),
          LLX::util::VectorGetX(m.c[2]), LLX::util::VectorGetX(m.c[3]));
  out << buf;

  sprintf(buf, "\t%10.6f %10.6f %10.6f %10.6f\n",
          LLX::util::VectorGetY(m.c[0]), LLX::util::VectorGetY(m.c[1]),
          LLX::util::VectorGetY(m.c[2]), LLX::util::VectorGetY(m.c[3]));
  out << buf;

  sprintf(buf, "\t%10.6f %10.6f %10.6f %10.6f\n",
          LLX::util::VectorGetZ(m.c[0]), LLX::util::VectorGetZ(m.c[1]),
          LLX::util::VectorGetZ(m.c[2]), LLX::util::VectorGetZ(m.c[3]));
  out << buf;

  sprintf(buf, "\t%10.6f %10.6f %10.6f %10.6f",
          LLX::util::VectorGetW(m.c[0]), LLX::util::VectorGetW(m.c[1]),
          LLX::util::VectorGetW(m.c[2]), LLX::util::VectorGetW(m.c[3]));
  out << buf;

  return out;
}

#endif // _LLX_UTIL_MATRIX_MATRIX_DEBUG_H_