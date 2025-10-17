/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#ifndef RUBY_H
#define RUBY_H 1

#define HAVE_RUBY_DEFINES_H     1
#define HAVE_RUBY_ENCODING_H    1
#define HAVE_RUBY_INTERN_H      1
#define HAVE_RUBY_IO_H          1
#define HAVE_RUBY_MISSING_H     1
#define HAVE_RUBY_ONIGURUMA_H   1
#define HAVE_RUBY_RE_H          1
#define HAVE_RUBY_REGEX_H       1
#define HAVE_RUBY_RUBY_H        1
#define HAVE_RUBY_ST_H          1
#define HAVE_RUBY_THREAD_H      1
#define HAVE_RUBY_UTIL_H        1
#define HAVE_RUBY_VERSION_H     1
#define HAVE_RUBY_VM_H          1
#ifdef _WIN32
#define HAVE_RUBY_WIN32_H       1
#endif

#include "ruby/ruby.h"

#endif /* RUBY_H */
