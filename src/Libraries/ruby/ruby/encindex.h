/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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
#ifndef RUBY_ENCINDEX_H
#define RUBY_ENCINDEX_H 1
#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

enum ruby_preserved_encindex {
    RUBY_ENCINDEX_ASCII,
    RUBY_ENCINDEX_UTF_8,
    RUBY_ENCINDEX_US_ASCII,

    /* preserved indexes */
    RUBY_ENCINDEX_UTF_16BE,
    RUBY_ENCINDEX_UTF_16LE,
    RUBY_ENCINDEX_UTF_32BE,
    RUBY_ENCINDEX_UTF_32LE,
    RUBY_ENCINDEX_UTF_16,
    RUBY_ENCINDEX_UTF_32,
    RUBY_ENCINDEX_UTF8_MAC,

    /* for old options of regexp */
    RUBY_ENCINDEX_EUC_JP,
    RUBY_ENCINDEX_Windows_31J,

    RUBY_ENCINDEX_BUILTIN_MAX
};

#define ENCINDEX_ASCII       RUBY_ENCINDEX_ASCII
#define ENCINDEX_UTF_8       RUBY_ENCINDEX_UTF_8
#define ENCINDEX_US_ASCII    RUBY_ENCINDEX_US_ASCII
#define ENCINDEX_UTF_16BE    RUBY_ENCINDEX_UTF_16BE
#define ENCINDEX_UTF_16LE    RUBY_ENCINDEX_UTF_16LE
#define ENCINDEX_UTF_32BE    RUBY_ENCINDEX_UTF_32BE
#define ENCINDEX_UTF_32LE    RUBY_ENCINDEX_UTF_32LE
#define ENCINDEX_UTF_16      RUBY_ENCINDEX_UTF_16
#define ENCINDEX_UTF_32      RUBY_ENCINDEX_UTF_32
#define ENCINDEX_UTF8_MAC    RUBY_ENCINDEX_UTF8_MAC
#define ENCINDEX_EUC_JP      RUBY_ENCINDEX_EUC_JP
#define ENCINDEX_Windows_31J RUBY_ENCINDEX_Windows_31J
#define ENCINDEX_BUILTIN_MAX RUBY_ENCINDEX_BUILTIN_MAX

#define rb_ascii8bit_encindex() RUBY_ENCINDEX_ASCII
#define rb_utf8_encindex()      RUBY_ENCINDEX_UTF_8
#define rb_usascii_encindex()   RUBY_ENCINDEX_US_ASCII

int rb_enc_find_index2(const char *name, long len);

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* RUBY_ENCINDEX_H */
