/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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

#include <wtf/NeverDestroyed.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

#define WEBCORE_COMMON_ATOM_STRINGS_FOR_EACH_KEYWORD(macro) \
    macro(all, "all") \
    macro(alternative, "alternative") \
    macro(applicationXHTMLContentType, "application/xhtml+xml") \
    macro(applicationXMLContentType, "application/xml") \
    macro(applicationOctetStream, "application/octet-stream") \
    macro(auto, "auto") \
    macro(captions, "captions") \
    macro(commentary, "commentary") \
    macro(cssContentType, "text/css") \
    macro(eager, "eager") \
    macro(email, "email") \
    macro(false, "false") \
    macro(imageSVGContentType, "image/svg+xml") \
    macro(lazy, "lazy") \
    macro(main, "main") \
    macro(manual, "manual") \
    macro(none, "none") \
    macro(off, "off") \
    macro(on, "on") \
    macro(plaintextOnly, "plaintext-only") \
    macro(print, "print") \
    macro(reset, "reset") \
    macro(screen, "screen") \
    macro(search, "search") \
    macro(star, "*") \
    macro(submit, "submit") \
    macro(subtitles, "subtitles") \
    macro(tel, "tel") \
    macro(text, "text") \
    macro(textHTMLContentType, "text/html") \
    macro(textPlainContentType, "text/plain") \
    macro(textXMLContentType, "text/xml") \
    macro(true, "true") \
    macro(url, "url") \
    macro(xml, "xml") \
    macro(xmlns, "xmlns")


#define DECLARE_COMMON_ATOM(atomName, atomValue) \
    extern MainThreadLazyNeverDestroyed<const AtomString> atomName ## AtomData; \
    inline const AtomString& atomName ## Atom() { return atomName ## AtomData.get(); }

WEBCORE_COMMON_ATOM_STRINGS_FOR_EACH_KEYWORD(DECLARE_COMMON_ATOM)

#undef DECLARE_COMMON_ATOM

WEBCORE_EXPORT void initializeCommonAtomStrings();

} // namespace WebCore
