/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
#ifndef __CLASS_FACTORY__
#define __CLASS_FACTORY__

#include    "StdAfx.h"

//===========================================================================================================================
//	ClassFactory
//===========================================================================================================================

class ClassFactory : public IClassFactory
{
protected:

DWORD mRefCount;
CLSID mCLSIDObject;

public:

ClassFactory( CLSID inCLSID );
~ClassFactory( void );

// IUnknown methods

STDMETHODIMP            QueryInterface( REFIID inID, LPVOID *outResult );
STDMETHODIMP_( DWORD )  AddRef( void );
STDMETHODIMP_( DWORD )  Release( void );

// IClassFactory methods

STDMETHODIMP    CreateInstance( LPUNKNOWN inUnknown, REFIID inID, LPVOID *outObject );
STDMETHODIMP    LockServer( BOOL inLock );
};

#endif  // __CLASS_FACTORY__
