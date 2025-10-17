/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 21, 2023.
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
#ifndef _OBJC_OBJCDT_JSON_H_
#define _OBJC_OBJCDT_JSON_H_

#include <cstdint>
#include <cstdbool>
#include <stdio.h>
#include <functional>

namespace json {

enum context: uint8_t {
    root,
    array_value,
    object_value,
    object_key,
    done,
};

class writer {
private:
    FILE *_file;
    context _context;
    int _depth;
    bool _needs_comma;

    void begin_value(int sep = '\0');
    void advance(context old);
    void key(const char *key);

public:

    writer(FILE *f);
    ~writer();

    void object(std::function<void()>);
    void object(const char *key, std::function<void()>);

    void array(std::function<void()>);
    void array(const char *key, std::function<void()>);

    void boolean(bool value);
    void boolean(const char *key, bool value);

    void number(uint64_t value);
    void number(const char *key, uint64_t value);

    void string(const char *s);
    void string(const char *key, const char *s);

    __printflike(2, 3)
    void stringf(const char *fmt, ...);

    __printflike(3, 4)
    void stringf(const char *key, const char *fmt, ...);
};

}

#endif /* _OBJC_OBJCDT_JSON_H_ */
