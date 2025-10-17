/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
/****************************************************************************
 *  Author: Thomas E. Dickey                                                *
 ****************************************************************************/

/*
 * $Id: tparm_type.h,v 1.1 2014/05/21 16:57:56 tom Exp $
 *
 * determine expected/actual number of parameters to setup for tparm
 */
#ifndef TPARM_TYPE_H
#define TPARM_TYPE_H 1

#define USE_LIBTINFO
#include <progs.priv.h>

typedef enum {
    Other = -1
    ,Str
    ,Numbers = 0
    ,Num_Str
    ,Num_Str_Str
    ,Str_Str
} TParams;

extern TParams tparm_type(const char *name);

#endif /* TPARM_TYPE_H */
