/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#if !defined(NUMBER_H_INCLUDED_)
#define NUMBER_H_INCLUDED_

#include "object.h"
#include "sequence.h"

namespace py {
    
class number : public object
{
public:
  number() : object() {};
  number(int i) : object(i) {};
  number(long i) : object(i) {};
  number(unsigned long i) : object(i) {};
  number(double d) : object(d) {}
  number(std::complex<double>& d) : object(d) {};

  number(const number& other) : object(other) {};
  number(PyObject* obj) : object(obj) {
    _violentTypeCheck();
  };
  virtual ~number() {};

  virtual number& operator=(const number& other) {
    grab_ref(other);
    return *this;
  };
  /*virtual*/ number& operator=(const object& other) {
    grab_ref(other);
    _violentTypeCheck();
    return *this;
  };
  virtual void _violentTypeCheck() {
    if (!PyNumber_Check(_obj)) {
      grab_ref(0);
      fail(PyExc_TypeError, "Not a number");
    }
  };
  //PyNumber_Absolute
  number abs() const {
    PyObject* rslt = PyNumber_Absolute(_obj);
    if (rslt==0)
      fail(PyExc_TypeError, "failed to get absolute value");
    return lose_ref(rslt);
  };
  //PyNumber_Add
  number operator+(const number& rhs) const {
    PyObject*  rslt = PyNumber_Add(_obj, rhs);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper rhs for +");
    return lose_ref(rslt);
  };
  //PyNumber_And
  number operator&(const number& rhs) const {
    PyObject*  rslt = PyNumber_And(_obj, rhs);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper rhs for &");
    return lose_ref(rslt);
  };
  //PyNumber_Coerce
  //PyNumber_Divide
  number operator/(const number& rhs) const {
    PyObject*  rslt = PyNumber_Divide(_obj, rhs);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper rhs for /");
    return lose_ref(rslt);
  };
  //PyNumber_Divmod
  sequence divmod(const number& rhs) const {
    PyObject*  rslt = PyNumber_Divmod(_obj, rhs);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper rhs for divmod");
    return lose_ref(rslt);
  };
  //PyNumber_Float
    operator double () const {
  PyObject*  F = PyNumber_Float(_obj);
  if (F==0)
      fail(PyExc_TypeError, "Cannot convert to double");
  double r = PyFloat_AS_DOUBLE(F);
  Py_DECREF(F);
  return r;
    };
  operator float () const {
    double rslt = (double) *this;
    //if (rslt > INT_MAX)
    //  fail(PyExc_TypeError, "Cannot convert to a float");
    return (float) rslt;
  };
  //PyNumber_Int
    operator long () const {
  PyObject*  Int = PyNumber_Int(_obj);
  if (Int==0)
      fail(PyExc_TypeError, "Cannot convert to long");
  long r = PyInt_AS_LONG(Int);
  Py_DECREF(Int);
  return r;
    };
  operator int () const {
    long rslt = (long) *this;
    if (rslt > INT_MAX)
      fail(PyExc_TypeError, "Cannot convert to an int");
    return (int) rslt;
  };
  //PyNumber_Invert
  number operator~ () const {
    PyObject* rslt = PyNumber_Invert(_obj);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper type for ~");
    return lose_ref(rslt);
  };
  //PyNumber_Long
  //PyNumber_Lshift
  number operator<<(const number& rhs) const {
    PyObject*  rslt = PyNumber_Lshift(_obj, rhs);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper rhs for <<");
    return lose_ref(rslt);
  };
  //PyNumber_Multiply
  number operator*(const number& rhs) const {
    PyObject*  rslt = PyNumber_Multiply(_obj, rhs);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper rhs for *");
    return lose_ref(rslt);
  };
  //PyNumber_Negative
  number operator- () const {
    PyObject* rslt = PyNumber_Negative(_obj);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper type for unary -");
    return lose_ref(rslt);
  };
  //PyNumber_Or
  number operator|(const number& rhs) const {
    PyObject*  rslt = PyNumber_Or(_obj, rhs);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper rhs for |");
    return lose_ref(rslt);
  };
  //PyNumber_Positive
  number operator+ () const {
    PyObject* rslt = PyNumber_Positive(_obj);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper type for unary +");
    return lose_ref(rslt);
  };
  //PyNumber_Remainder
  number operator%(const number& rhs) const {
    PyObject*  rslt = PyNumber_Remainder(_obj, rhs);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper rhs for %");
    return lose_ref(rslt);
  };
  //PyNumber_Rshift
  number operator>>(const number& rhs) const {
    PyObject*  rslt = PyNumber_Rshift(_obj, rhs);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper rhs for >>");
    return lose_ref(rslt);
  };
  //PyNumber_Subtract
  number operator-(const number& rhs) const {
    PyObject*  rslt = PyNumber_Subtract(_obj, rhs);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper rhs for -");
    return lose_ref(rslt);
  };
  //PyNumber_Xor
  number operator^(const number& rhs) const {
    PyObject*  rslt = PyNumber_Xor(_obj, rhs);
    if (rslt==0)
      fail(PyExc_TypeError, "Improper rhs for ^");
    return lose_ref(rslt);
  };
};

} // namespace py

#endif //NUMBER_H_INCLUDED_
