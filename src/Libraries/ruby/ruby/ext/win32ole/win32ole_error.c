/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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

#include "win32ole.h"

static VALUE ole_hresult2msg(HRESULT hr);

static VALUE
ole_hresult2msg(HRESULT hr)
{
    VALUE msg = Qnil;
    char *p_msg = NULL;
    char *term = NULL;
    DWORD dwCount;

    char strhr[100];
    sprintf(strhr, "    HRESULT error code:0x%08x\n      ", (unsigned)hr);
    msg = rb_str_new2(strhr);
    dwCount = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                            FORMAT_MESSAGE_FROM_SYSTEM |
                            FORMAT_MESSAGE_IGNORE_INSERTS,
                            NULL, hr,
                            MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US),
                            (LPTSTR)&p_msg, 0, NULL);
    if (dwCount == 0) {
        dwCount = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                                FORMAT_MESSAGE_FROM_SYSTEM |
                                FORMAT_MESSAGE_IGNORE_INSERTS,
                                NULL, hr, cWIN32OLE_lcid,
                                (LPTSTR)&p_msg, 0, NULL);
    }
    if (dwCount > 0) {
        term = p_msg + strlen(p_msg);
        while (p_msg < term) {
            term--;
            if (*term == '\r' || *term == '\n')
                *term = '\0';
            else break;
        }
        if (p_msg[0] != '\0') {
            rb_str_cat2(msg, p_msg);
        }
    }
    LocalFree(p_msg);
    return msg;
}

void
ole_raise(HRESULT hr, VALUE ecs, const char *fmt, ...)
{
    va_list args;
    VALUE msg;
    VALUE err_msg;
    va_init_list(args, fmt);
    msg = rb_vsprintf(fmt, args);
    va_end(args);

    err_msg = ole_hresult2msg(hr);
    if(err_msg != Qnil) {
        rb_str_cat2(msg, "\n");
        rb_str_append(msg, err_msg);
    }
    rb_exc_raise(rb_exc_new_str(ecs, msg));
}

VALUE eWIN32OLERuntimeError;
VALUE eWIN32OLEQueryInterfaceError;

void
Init_win32ole_error(void)
{
    /*
     * Document-class: WIN32OLERuntimeError
     *
     * Raised when OLE processing failed.
     *
     * EX:
     *
     *   obj = WIN32OLE.new("NonExistProgID")
     *
     * raises the exception:
     *
     *   WIN32OLERuntimeError: unknown OLE server: `NonExistProgID'
     *       HRESULT error code:0x800401f3
     *         Invalid class string
     *
     */
    eWIN32OLERuntimeError = rb_define_class("WIN32OLERuntimeError", rb_eRuntimeError);
    eWIN32OLEQueryInterfaceError = rb_define_class("WIN32OLEQueryInterfaceError", eWIN32OLERuntimeError);
}
