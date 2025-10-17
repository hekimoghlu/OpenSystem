/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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
#include "Tfp_Arrays.h"

Tfp_ArrayType       
*Tfp_ArrayInit( Tfp_ArrayDeleteProc *cleanProc )
{
    Tfp_ArrayType   *arr;
    
    arr = (Tfp_ArrayType *) Tcl_Alloc( sizeof(Tfp_ArrayType) );
    arr->table = (Tcl_HashTable *) Tcl_Alloc( sizeof(Tcl_HashTable) );
    Tcl_InitHashTable( arr->table, TCL_STRING_KEYS );
    arr->cleanProc = cleanProc;
    return arr;
}

void                
Tfp_ArrayDestroy( Tfp_ArrayType *arr )
{
    Tcl_HashEntry   *p;
    Tcl_HashSearch  s;
    
    if (arr->cleanProc != (Tfp_ArrayDeleteProc *) NULL) {
        for (p = Tcl_FirstHashEntry( arr->table, &s ); p != (Tcl_HashEntry *) NULL;
                p = Tcl_NextHashEntry( &s )) {
            (*arr->cleanProc) ( Tcl_GetHashValue( p ) );
        }
    }
    Tcl_DeleteHashTable( arr->table );
    Tcl_Free( (char *) arr->table );
    Tcl_Free( (char *) arr );
}

int                 
Tfp_ArrayGet( Tfp_ArrayType *arr, char *key, ClientData *returnValue )
{
    Tcl_HashEntry   *p;
    
    p = Tcl_FindHashEntry( arr->table, key );
    if (p == (Tcl_HashEntry *) NULL) {
        return 0;
    }
    *returnValue = Tcl_GetHashValue( p );
    return 1;
}

void                 
Tfp_ArraySet( Tfp_ArrayType *arr, char *key, ClientData value )
{
    int             junk;
    Tcl_HashEntry   *p;
    
    p = Tcl_CreateHashEntry( arr->table, key, &junk );
    Tcl_SetHashValue( p, value );
}

void                
Tfp_ArrayDelete( Tfp_ArrayType *arr, char *key )
{
    Tcl_HashEntry   *p;
    
    p = Tcl_FindHashEntry( arr->table, key );
    if (p == (Tcl_HashEntry *) NULL) {
        return;
    }    
    (*arr->cleanProc) ( Tcl_GetHashValue( p ) );
    Tcl_DeleteHashEntry( p );
}

/*---------------------------------------------------------------------------*/