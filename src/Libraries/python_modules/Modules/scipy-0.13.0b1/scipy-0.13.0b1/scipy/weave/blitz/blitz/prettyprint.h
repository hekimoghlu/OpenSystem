/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 23, 2023.
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

// -*- C++ -*-
/***************************************************************************
 * blitz/prettyprint.h      Format object for pretty-printing of
 *                          array expressions
 *
 * $Id: prettyprint.h 1414 2005-11-01 22:04:59Z cookedm $
 *
 * Copyright (C) 1997-2001 Todd Veldhuizen <tveldhui@oonumerics.org>
 *
 * This code was relicensed under the modified BSD license for use in SciPy
 * by Todd Veldhuizen (see LICENSE.txt in the weave directory).
 *
 *
 * Suggestions:          blitz-dev@oonumerics.org
 * Bugs:                 blitz-bugs@oonumerics.org
 *
 * For more information, please see the Blitz++ Home Page:
 *    http://oonumerics.org/blitz/
 *
 ***************************************************************************/

#ifndef BZ_PRETTYPRINT_H
#define BZ_PRETTYPRINT_H

BZ_NAMESPACE(blitz)

class prettyPrintFormat {

public:
    prettyPrintFormat(const bool terse = false)
        : tersePrintingSelected_(terse) 
    {
        arrayOperandCounter_ = 0;
        scalarOperandCounter_ = 0;
        dumpArrayShapes_ = false;
    }

    void setDumpArrayShapesMode()  { dumpArrayShapes_ = true; }
    char nextArrayOperandSymbol()  
    { 
        return static_cast<char>('A' + ((arrayOperandCounter_++) % 26)); 
    }
    char nextScalarOperandSymbol() 
    { 
        return static_cast<char>('s' + ((scalarOperandCounter_++) % 26)); 
    }

    bool tersePrintingSelected() const { return tersePrintingSelected_; }
    bool dumpArrayShapesMode()   const { return dumpArrayShapes_; }

private:
    bool tersePrintingSelected_;
    bool dumpArrayShapes_;
    int arrayOperandCounter_;
    int scalarOperandCounter_;
};

BZ_NAMESPACE_END

#endif // BZ_PRETTYPRINT_H
