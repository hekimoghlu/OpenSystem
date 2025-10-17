/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#include <fstream>
#include <iomanip>
#include <iostream>

class bzofstream : public std::ofstream {

public:
    bzofstream(const char* filename, const char* description,
        const char* sourceFile, const char* mnemonic)
        : std::ofstream(filename)
    {
        (*this) << 
"/***************************************************************************\n"
" * blitz/" << filename << "\t" << description << std::endl <<
" *\n"
" * This code was relicensed under the modified BSD license for use in SciPy\n"
" * by Todd Veldhuizen (see LICENSE.txt in the weave directory).\n"
" *\n"
" *\n"
" * Suggestions:          blitz-suggest@cybervision.com\n"
" * Bugs:                 blitz-bugs@cybervision.com\n"
" *\n"
" * For more information, please see the Blitz++ Home Page:\n"
" *    http://seurat.uwaterloo.ca/blitz/\n"
" *\n"
" ***************************************************************************\n"
" *\n"
" */ " 
       << std::endl << std::endl
       << "// Generated source file.  Do not edit. " << std::endl
       << "// " << sourceFile << " " << __DATE__ << " " << __TIME__ 
       << std::endl << std::endl
       << "#ifndef " << mnemonic << std::endl
       << "#define " << mnemonic << std::endl << std::endl;
    }

    void include(const char* filename)
    {
        (*this) << "#include <blitz/" << filename << ">" << std::endl;
    }

    void beginNamespace()
    {
        (*this) << "BZ_NAMESPACE(blitz)" << std::endl << std::endl;
    }

    ~bzofstream()
    {
        (*this) << "BZ_NAMESPACE_END" << std::endl << std::endl
                << "#endif" << std::endl;
    }

};

