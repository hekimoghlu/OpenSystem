/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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

//
//  lib_newfs_msdos.c
//  newfs_msdos
//
//  Created by Kujan Lauz on 04/09/2022.
//

#include <stdio.h>
#include "lib_newfs_msdos.h"


lib_newfs_ctx_t newfs_ctx;

void newfs_set_context_properties(newfs_msdos_print_funct_t print,
                                  newfs_msdos_wipefs_func_t wipefs,
                                  newfs_client_ctx_t client)
{
    newfs_ctx.wipefs = wipefs;
    newfs_ctx.print = print;
    newfs_ctx.client_ctx = client;
}

void newfs_print(lib_newfs_ctx_t c, int level, const char *fmt, ...)
{
    if (c.print) {
        va_list ap;
        va_start(ap, fmt);
        c.print(c.client_ctx, level, fmt, ap);
        va_end(ap);
    }
}

newfs_msdos_wipefs_func_t newfs_get_wipefs_function_callback(void) {
    return newfs_ctx.wipefs;
}

void newfs_set_wipefs_function_callback(newfs_msdos_wipefs_func_t func) {
    newfs_ctx.wipefs = func;
}

newfs_msdos_print_funct_t newfs_get_print_function_callback(void) {
    return newfs_ctx.print;
}

void newfs_set_print_function_callback(newfs_msdos_print_funct_t func) {
    newfs_ctx.print = func;
}

newfs_client_ctx_t newfs_get_client(void) {
    return newfs_ctx.client_ctx;
}

void newfs_set_client (newfs_client_ctx_t c) {
    newfs_ctx.client_ctx = c;
}
