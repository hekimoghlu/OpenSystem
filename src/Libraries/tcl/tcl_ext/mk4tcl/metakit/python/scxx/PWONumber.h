/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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
#if !defined(PWONUMBER_H_INCLUDED_)
#define PWONUMBER_H_INCLUDED_

#include "PWOBase.h"
#include "PWOSequence.h"

#if defined(PY_LONG_LONG) && !defined(LONG_LONG)
#define LONG_LONG PY_LONG_LONG
#endif 

class PWONumber: public PWOBase {
  public:
    PWONumber(): PWOBase(){}
    ;
    PWONumber(int i): PWOBase(PyInt_FromLong(i)) {
        LoseRef(_obj);
    }
    PWONumber(long i): PWOBase(PyInt_FromLong(i)) {
        LoseRef(_obj);
    }
    PWONumber(unsigned long i): PWOBase(PyLong_FromUnsignedLong(i)) {
        LoseRef(_obj);
    }
#ifdef HAVE_LONG_LONG
    PWONumber(LONG_LONG i): PWOBase(PyLong_FromLongLong(i)) {
        LoseRef(_obj);
    }
#endif 
    PWONumber(double d): PWOBase(PyFloat_FromDouble(d)) {
        LoseRef(_obj);
    }

    PWONumber(const PWONumber &other): PWOBase(other){}
    ;
    PWONumber(PyObject *obj): PWOBase(obj) {
        _violentTypeCheck();
    };
    virtual ~PWONumber(){}
    ;

    virtual PWONumber &operator = (const PWONumber &other) {
        GrabRef(other);
        return  *this;
    };
     /*virtual*/PWONumber &operator = (const PWOBase &other) {
        GrabRef(other);
        _violentTypeCheck();
        return  *this;
    };
    virtual void _violentTypeCheck() {
        if (!PyNumber_Check(_obj)) {
            GrabRef(0);
            Fail(PyExc_TypeError, "not a number");
        }
    };
    //PyNumber_Absolute
    PWONumber abs()const {
        PyObject *rslt = PyNumber_Absolute(_obj);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Failed to get absolute value");
        return LoseRef(rslt);
    };
    //PyNumber_Add
    PWONumber operator + (const PWONumber &rhs)const {
        PyObject *rslt = PyNumber_Add(_obj, rhs);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper rhs for +");
        return LoseRef(rslt);
    };
    //PyNumber_And
    PWONumber operator &(const PWONumber &rhs)const {
        PyObject *rslt = PyNumber_And(_obj, rhs);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper rhs for &");
        return LoseRef(rslt);
    };
    //PyNumber_Coerce
    //PyNumber_Divide
    PWONumber operator / (const PWONumber &rhs)const {
        PyObject *rslt = PyNumber_Divide(_obj, rhs);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper rhs for /");
        return LoseRef(rslt);
    };
    //PyNumber_Divmod
    PWOSequence divmod(const PWONumber &rhs)const {
        PyObject *rslt = PyNumber_Divmod(_obj, rhs);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper rhs for divmod");
        return LoseRef(rslt);
    };
    //PyNumber_Float
    operator double()const {
        PyObject *F = PyNumber_Float(_obj);
        if (F == NULL)
          Fail(PyExc_TypeError, "Cannot convert to double");
        double r = PyFloat_AS_DOUBLE(F);
        Py_DECREF(F);
        return r;
    };
    /* // no easy, safe way to do this 
    operator float () const {
    double rslt = (double) *this;
    return (float) rslt;
    }; */
    //PyNumber_Int
    operator long()const {
        PyObject *Int = PyNumber_Int(_obj);
        if (Int == NULL)
          Fail(PyExc_TypeError, "can't convert to int");
        long r = PyInt_AsLong(_obj);
        if (r ==  - 1)
          FailIfPyErr();
        return r;
    };
    operator int()const {
        long rslt = (long) *this;
        if (rslt > INT_MAX)
          Fail(PyExc_ValueError, "int too large to convert to C int");
        return (int)rslt;
    };
    //PyNumber_Invert
    PWONumber operator ~()const {
        PyObject *rslt = PyNumber_Invert(_obj);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper type for ~");
        return LoseRef(rslt);
    };
    //PyNumber_Long
#ifdef HAVE_LONG_LONG
    operator LONG_LONG()const {
        PyObject *Long = PyNumber_Long(_obj);
        if (Long == NULL)
          Fail(PyExc_TypeError, "can't convert to long int");
        LONG_LONG r = PyLong_AsLongLong(Long);
        if (r ==  - 1 && PyErr_Occurred() != NULL)
          Fail(PyExc_ValueError, "long int too large to convert to C long long")
            ;
        Py_DECREF(Long);
        return r;
    };
#endif 
    //PyNumber_Lshift
    PWONumber operator << (const PWONumber &rhs)const {
        PyObject *rslt = PyNumber_Lshift(_obj, rhs);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper rhs for <<");
        return LoseRef(rslt);
    };
    //PyNumber_Multiply
    PWONumber operator *(const PWONumber &rhs)const {
        PyObject *rslt = PyNumber_Multiply(_obj, rhs);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper rhs for *");
        return LoseRef(rslt);
    };
    //PyNumber_Negative
    PWONumber operator - ()const {
        PyObject *rslt = PyNumber_Negative(_obj);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper type for unary -");
        return LoseRef(rslt);
    };
    //PyNumber_Or
    PWONumber operator | (const PWONumber &rhs)const {
        PyObject *rslt = PyNumber_Or(_obj, rhs);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper rhs for |");
        return LoseRef(rslt);
    };
    //PyNumber_Positive
    PWONumber operator + ()const {
        PyObject *rslt = PyNumber_Positive(_obj);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper type for unary +");
        return LoseRef(rslt);
    };
    //PyNumber_Remainder
    PWONumber operator % (const PWONumber &rhs)const {
        PyObject *rslt = PyNumber_Remainder(_obj, rhs);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper rhs for %");
        return LoseRef(rslt);
    };
    //PyNumber_Rshift
    PWONumber operator >> (const PWONumber &rhs)const {
        PyObject *rslt = PyNumber_Rshift(_obj, rhs);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper rhs for >>");
        return LoseRef(rslt);
    };
    //PyNumber_Subtract
    PWONumber operator - (const PWONumber &rhs)const {
        PyObject *rslt = PyNumber_Subtract(_obj, rhs);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper rhs for -");
        return LoseRef(rslt);
    };
    //PyNumber_Xor
    PWONumber operator ^ (const PWONumber &rhs)const {
        PyObject *rslt = PyNumber_Xor(_obj, rhs);
        if (rslt == NULL)
          Fail(PyExc_TypeError, "Improper rhs for ^");
        return LoseRef(rslt);
    };
};

#endif //PWONUMBER_H_INCLUDED_
