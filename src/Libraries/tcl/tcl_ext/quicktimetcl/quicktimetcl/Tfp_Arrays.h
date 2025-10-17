/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
#include "tcl.h"

typedef void        (Tfp_ArrayDeleteProc) (ClientData);

typedef struct {
    Tcl_HashTable       *table;
    Tfp_ArrayDeleteProc *cleanProc;
} Tfp_ArrayType;

Tfp_ArrayType       *Tfp_ArrayInit( Tfp_ArrayDeleteProc *cleanProc );
void                Tfp_ArrayDestroy( Tfp_ArrayType *arr );
int                 Tfp_ArrayGet( Tfp_ArrayType *arr, char *key, ClientData *returnValue );
void                Tfp_ArraySet( Tfp_ArrayType *arr, char *key, ClientData value );
void                Tfp_ArrayDelete( Tfp_ArrayType *arr, char *key );


