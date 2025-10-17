/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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

#include <ruby.h>
#include <ruby/encoding.h>
#include <windows.h>
#ifndef NTDDI_VERSION
#define NTDDI_VERSION 0x06000000
#endif
#include <iphlpapi.h>

static VALUE
w32error_make_error(DWORD e)
{
    VALUE code = ULONG2NUM(e);
    return rb_class_new_instance(1, &code, rb_path2class("Win32::Resolv::Error"));
}

static void
w32error_raise(DWORD e)
{
    rb_exc_raise(w32error_make_error(e));
}

static VALUE
get_dns_server_list(VALUE self)
{
    FIXED_INFO *fixedinfo = NULL;
    ULONG buflen = 0;
    DWORD ret;
    VALUE buf, nameservers = Qnil;

    ret = GetNetworkParams(NULL, &buflen);
    if (ret != NO_ERROR && ret != ERROR_BUFFER_OVERFLOW) {
	w32error_raise(ret);
    }
    fixedinfo = ALLOCV(buf, buflen);
    ret = GetNetworkParams(fixedinfo, &buflen);
    if (ret == NO_ERROR) {
	const IP_ADDR_STRING *ipaddr = &fixedinfo->DnsServerList;
	nameservers = rb_ary_new();
	do {
	    const char *s = ipaddr->IpAddress.String;
	    if (!*s) continue;
	    if (strcmp(s, "0.0.0.0") == 0) continue;
	    rb_ary_push(nameservers, rb_str_new_cstr(s));
	} while ((ipaddr = ipaddr->Next) != NULL);
    }
    ALLOCV_END(buf);
    if (ret != NO_ERROR) w32error_raise(ret);

    return nameservers;
}

void
InitVM_resolv(void)
{
    VALUE mWin32 = rb_define_module("Win32");
    VALUE resolv = rb_define_module_under(mWin32, "Resolv");
    VALUE singl = rb_singleton_class(resolv);
    rb_define_private_method(singl, "get_dns_server_list", get_dns_server_list, 0);
}

void
Init_resolv(void)
{
    InitVM(resolv);
}
