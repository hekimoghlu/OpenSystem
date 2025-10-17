/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#include <assert.h>
#include "json.h"

namespace json {

static bool
context_is_value(context c)
{
    return c == root || c == array_value || c == object_value;
}

writer::writer(FILE *f)
: _file(f)
, _context(root)
, _depth(0)
, _needs_comma(false)
{
}

writer::~writer()
{
    fputc('\n', _file);
    fflush(_file);
}

void
writer::begin_value(int sep)
{
    if (_needs_comma) {
        _needs_comma = false;
        if (sep) {
            fprintf(_file, ", %c\n", sep);
            return;
        }
        fputs(",\n", _file);
    }
    if (_context == array_value || _context == object_key) {
        fprintf(_file, "%*s", _depth * 2, "");
    }
    if (sep) {
        fprintf(_file, "%c\n", sep);
    }
}

void
writer::advance(context c)
{
    switch (c) {
    case root:
        _context = done;
        _needs_comma = false;
        break;
    case array_value:
        _context = array_value;
        _needs_comma = true;
        break;
    case object_value:
        _context = object_key;
        _needs_comma = true;
        break;
    case object_key:
        _context = object_value;
        _needs_comma = false;
        break;
    case done:
        assert(false);
        break;
    }
}

void
writer::key(const char *key)
{
    assert(_context == object_key);

    begin_value();
    fprintf(_file, "\"%s\": ", key);
    advance(_context);
}

void
writer::object(std::function<void()> f)
{
    context old = _context;
    assert(context_is_value(old));

    begin_value('{');

    _depth++;
    _context = object_key;
    _needs_comma = false;
    f();

    _depth--;
    fprintf(_file, "\n%*s}", _depth * 2, "");
    advance(old);
}

void
writer::object(const char *k, std::function<void()> f)
{
    key(k);
    object(f);
}

void
writer::array(std::function<void()> f)
{
    context old = _context;
    assert(context_is_value(old));

    begin_value('[');

    _depth++;
    _context = array_value;
    _needs_comma = false;
    f();

    _depth--;
    fprintf(_file, "\n%*s]", _depth * 2, "");
    advance(old);
}

void
writer::array(const char *k, std::function<void()> f)
{
    key(k);
    array(f);
}

void
writer::boolean(bool value)
{
    assert(context_is_value(_context));
    begin_value();
    fputs(value ? "true" : "false", _file);
    advance(_context);
}

void
writer::boolean(const char *k, bool value)
{
    key(k);
    boolean(value);
}

void
writer::number(uint64_t value)
{
    assert(context_is_value(_context));
    begin_value();
    fprintf(_file, "%lld", value);
    advance(_context);
}

void
writer::number(const char *k, uint64_t value)
{
    key(k);
    number(value);
}

void
writer::string(const char *s)
{
    assert(context_is_value(_context));
    begin_value();
    fprintf(_file, "\"%s\"", s);
    advance(_context);
}

void
writer::string(const char *k, const char *s)
{
    key(k);
    string(s);
}

void
writer::stringf(const char *fmt, ...)
{
    va_list ap;

    assert(context_is_value(_context));
    begin_value();
    fputc('"', _file);
    va_start(ap, fmt);
    vfprintf(_file, fmt, ap);
    va_end(ap);
    fputc('"', _file);
    advance(_context);
}

void
writer::stringf(const char *k, const char *fmt, ...)
{
    va_list ap;

    key(k);

    assert(context_is_value(_context));
    begin_value();
    fputc('"', _file);
    va_start(ap, fmt);
    vfprintf(_file, fmt, ap);
    va_end(ap);
    fputc('"', _file);
    advance(_context);
}

} // json
