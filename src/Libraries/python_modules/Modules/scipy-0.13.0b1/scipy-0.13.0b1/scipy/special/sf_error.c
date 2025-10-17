/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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

#include <stdlib.h>
#include <stdarg.h>

#include <Python.h>

#include "sf_error.h"

#if PY_VERSION_HEX < 0x02050000
  #define PyErr_WarnEx(category, message, stacklevel) PyErr_Warn(category, message)
#endif

const char *sf_error_messages[] = {
    "no error",
    "singularity",
    "underflow",
    "overflow",
    "too slow convergence",
    "loss of precision",
    "no result obtained",
    "domain error",
    "invalid input argument",
    "other error",
    NULL
};

static int print_error_messages = 0;

extern int wrap_PyUFunc_getfperr();

int sf_error_set_print(int flag)
{
    int old_flag = print_error_messages;
    print_error_messages = flag;
    return old_flag;
}

int sf_error_get_print()
{
    return print_error_messages;
}

void sf_error(char *func_name, sf_error_t code, char *fmt, ...)
{
    char msg[2048], info[1024];
    static PyObject *py_SpecialFunctionWarning = NULL;
    va_list ap;

    if (!print_error_messages) {
        return;
    }

    if (func_name == NULL) {
        func_name = "?";
    }

    if ((int)code < 0 || (int)code >= 10) {
        code = SF_ERROR_OTHER;
    }

    if (fmt != NULL && fmt[0] != '\0') {
        va_start(ap, fmt);
        PyOS_vsnprintf(info, 1024, fmt, ap);
        va_end(ap);
        PyOS_snprintf(msg, 2048, "scipy.special/%s: (%s) %s",
                      func_name, sf_error_messages[(int)code], info);
    }
    else {
        PyOS_snprintf(msg, 2048, "scipy.special/%s: %s",
                      func_name, sf_error_messages[(int)code]);
    }

    {
#ifdef WITH_THREAD
        PyGILState_STATE save = PyGILState_Ensure();
#endif

        if (PyErr_Occurred())
            goto skip_warn;

        if (py_SpecialFunctionWarning == NULL) {
            PyObject *scipy_special = NULL;

            scipy_special = PyImport_ImportModule("scipy.special");
            if (scipy_special == NULL) {
                PyErr_Clear();
                goto skip_warn;
            }

            py_SpecialFunctionWarning = PyObject_GetAttrString(
                scipy_special, "SpecialFunctionWarning");
            if (py_SpecialFunctionWarning == NULL) {
                PyErr_Clear();
                goto skip_warn;
            }
        }

        if (py_SpecialFunctionWarning != NULL) {
            PyErr_WarnEx(py_SpecialFunctionWarning, msg, 1);

            /*
             * The return value is ignored! We rely on the fact that the
             * Ufunc loop will call PyErr_Occurred() later on.
             */
        }

    skip_warn:
#ifdef WITH_THREAD
        PyGILState_Release(save);
#endif
    }
}

#define UFUNC_FPE_DIVIDEBYZERO  1
#define UFUNC_FPE_OVERFLOW      2
#define UFUNC_FPE_UNDERFLOW     4
#define UFUNC_FPE_INVALID       8

void sf_error_check_fpe(char *func_name)
{
    int status;
    status = wrap_PyUFunc_getfperr();
    if (status & UFUNC_FPE_DIVIDEBYZERO) {
        sf_error(func_name, SF_ERROR_SINGULAR, "floating point division by zero");
    }
    if (status & UFUNC_FPE_UNDERFLOW) {
        sf_error(func_name, SF_ERROR_UNDERFLOW, "floating point underflow");
    }
    if (status & UFUNC_FPE_OVERFLOW) {
        sf_error(func_name, SF_ERROR_OVERFLOW, "floating point overflow");
    }
    if (status & UFUNC_FPE_INVALID) {
        sf_error(func_name, SF_ERROR_DOMAIN, "floating point invalid value");
    }
}
