/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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
#if defined( __i386__ ) || defined( __x86_64__ )

#include <IOKit/IOTypes.h>
#include <architecture/i386/pio.h>  // x86 IN/OUT inline asm
#include "IOATARegI386.h"

OSDefineMetaClassAndAbstractStructors( IOATAReg8,  OSObject )
OSDefineMetaClassAndAbstractStructors( IOATAReg16, OSObject )
OSDefineMetaClassAndAbstractStructors( IOATAReg32, OSObject )

OSDefineMetaClassAndStructors( IOATAIOReg8,  IOATAReg8  )
OSDefineMetaClassAndStructors( IOATAIOReg16, IOATAReg16 )
OSDefineMetaClassAndStructors( IOATAIOReg32, IOATAReg32 )

#define ImplementIOATAIOReg(w, s)                             \
IOATAIOReg##w * IOATAIOReg##w::withAddress( UInt16 address )  \
{                                                             \
    IOATAIOReg##w * reg = new IOATAIOReg##w;                  \
                                                              \
    if ( reg && !reg->initWithAddress(address) )              \
    {                                                         \
        reg->release();                                       \
        reg = 0;                                              \
    }                                                         \
    return reg;                                               \
}                                                             \
                                                              \
bool IOATAIOReg##w::initWithAddress( UInt16 address )         \
{                                                             \
    if ( IOATAReg##w::init() == false ) return false;         \
    _address = address;                                       \
    return true;                                              \
}                                                             \
                                                              \
UInt16 IOATAIOReg##w::getAddress() const                      \
{                                                             \
    return _address;                                          \
}                                                             \
                                                              \
void IOATAIOReg##w::operator = (UInt##w rhs)                  \
{                                                             \
    out##s(_address, rhs);                                    \
}                                                             \
                                                              \
IOATAIOReg##w::operator UInt##w() const                       \
{                                                             \
    return in##s(_address);                                   \
}

ImplementIOATAIOReg( 8,  b )
ImplementIOATAIOReg( 16, w )
ImplementIOATAIOReg( 32, l )

#endif /* __i386__ || __x86_64__ */
