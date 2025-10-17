/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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
#pragma once

#include <stdio.h>
#include <wtf/PrintStream.h>

namespace WTF {

class FilePrintStream final : public PrintStream {
public:
    enum AdoptionMode {
        Adopt,
        Borrow
    };
    
    FilePrintStream(FILE*, AdoptionMode = Adopt);
    WTF_EXPORT_PRIVATE ~FilePrintStream() final;
    
    WTF_EXPORT_PRIVATE static std::unique_ptr<FilePrintStream> open(const char* filename, const char* mode);
    
    FILE* file() { return m_file; }
    
    void vprintf(const char* format, va_list) final WTF_ATTRIBUTE_PRINTF(2, 0);
    void flush() final;

private:
    FILE* m_file;
    AdoptionMode m_adoptionMode;
};

} // namespace WTF

using WTF::FilePrintStream;
