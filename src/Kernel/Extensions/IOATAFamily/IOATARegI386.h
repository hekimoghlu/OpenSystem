/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
#ifndef _IOATAREGI386_H
#define _IOATAREGI386_H

#include <libkern/c++/OSObject.h>

/*
 * IOATAReg: ATA register abstract base class.
 */
#define DefineIOATAReg(w)                                   \
class IOATAReg##w : public OSObject                         \
{                                                           \
    OSDeclareAbstractStructors( IOATAReg##w )               \
                                                            \
public:                                                     \
    virtual void operator = (UInt##w rhs) = 0;              \
    virtual operator UInt##w() const = 0;                   \
}

DefineIOATAReg( 8 );
DefineIOATAReg( 16 );
DefineIOATAReg( 32 );

typedef IOATAReg8  * IOATARegPtr8;
typedef IOATAReg16 * IOATARegPtr16;
typedef IOATAReg32 * IOATARegPtr32;

#define IOATARegPtr8Cast(x) (x)

/*
 * IOATAIOReg: I/O mapped ATA registers.
 */
#define DefineIOATAIOReg(w)                                 \
class IOATAIOReg##w : public IOATAReg##w                    \
{                                                           \
    OSDeclareDefaultStructors( IOATAIOReg##w )              \
                                                            \
protected:                                                  \
    UInt16 _address;                                        \
                                                            \
public:                                                     \
    static IOATAIOReg##w * withAddress( UInt16 address );   \
                                                            \
    virtual bool initWithAddress( UInt16 address );         \
    virtual UInt16 getAddress() const;                      \
                                                            \
    virtual void operator = (UInt##w rhs);                  \
    virtual operator UInt##w() const;                       \
}

DefineIOATAIOReg( 8 );
DefineIOATAIOReg( 16 );
DefineIOATAIOReg( 32 );

#endif /* !_IOATAREGI386_H */
