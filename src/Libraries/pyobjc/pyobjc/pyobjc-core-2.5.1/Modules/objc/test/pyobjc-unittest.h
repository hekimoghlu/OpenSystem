/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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

#ifndef PyObjC_UNITTEST_H
#define PyObjC_UNITTEST_H
/*!
 * @header pyobjc-unittest.h
 * @abstract Defining C-based unittests
 * @discussion
 *
 * This file defines a very simple unittest framework for the Objective-C
 * modules of PyObjC. 
 *
 * It is intended to be used with PyUnit, see objc.test.test_ctests for more
 * information on that.
 *
 * Usage:
 * <PRE>
 * BEGIN_UNITEST(IntSize)
 *     // This is a normal C block, starting with variable definitions
 *     int i = 3;
 *
 *     ASSERT_EQUALS(sizeof(int), sizeof(i), "%d");
 * 
 * END_UNITTEST
 * </PRE>
 *
 * And in the PyMethodDef list for the module:
 * <PRE>
 *     TESTDEF(IntSize),
 * </PRE>
 *
 * Use 'FAIL_IF' to abort a test due to a Python exception.
 */

#include <stdarg.h>

#define BEGIN_UNITTEST(name) \
	static PyObject* \
	test_##name (PyObject* self __attribute__((__unused__))) \
 	{  \

#define END_UNITTEST \
		Py_INCREF(Py_None); \
		return Py_None; \
	error: 			\
		return NULL;	\
	}			

#define FAIL_IF(expr) do { if ((expr)) goto error; } while(0)

#pragma GCC diagnostic push
#pragma clang diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#pragma clang diagnostic ignored "-Wformat-nonliteral"

static inline void
unittest_assert_failed(const char* file, int line, char* msg, ...)
{
	char buf[10240];
	va_list ap;

	va_start(ap, msg);
	vsnprintf(buf, sizeof(buf), msg, ap);
	va_end(ap);
	PyErr_Format(PyExc_AssertionError, "%s:%d %s", file, line, buf);
}

#pragma GCC diagnostic pop
#pragma clang diagnostic pop

#define ASSERT(expr) \
	do { \
		if (!(expr)) { \
			unittest_assert_failed(__FILE__, __LINE__,"%s",#expr); \
			goto error; \
		} \
	} while (0)

#define ASSERT_EQUALS(val1, val2, fmt) \
	do { \
		if ((val1) != (val2)) { \
			unittest_assert_failed(__FILE__, __LINE__, \
					fmt " != " fmt, (val1), (val2)); \
			goto error; \
		} \
	} while (0)

#define ASSERT_STREQUALS(val1, val2) \
	do { \
		const char* _val1 = (val1); \
		const char* _val2 = (val2); \
		\
		if (strcmp(_val1, _val2) != 0) { \
			unittest_assert_failed(__FILE__, __LINE__, \
					 "%s != %s", (_val1), (_val2)); \
			goto error; \
		} \
	} while (0)

#define ASSERT_ISINSTANCE(val, type) \
	do { \
		if (!Py##type##_Check((val))) { \
			unittest_assert_failed(__FILE__, __LINE__, \
				"type of value is %s not %s", \
				(val)->ob_type->tp_name, \
				Py##type##_Type.tp_name); \
			goto error; \
		} \
	} while (0)


#define TESTDEF(name) \
	{ \
		#name, \
		(PyCFunction)test_##name,  \
		METH_NOARGS,  \
		NULL \
	}

#endif /* PyObjC_UNITTEST_H */
