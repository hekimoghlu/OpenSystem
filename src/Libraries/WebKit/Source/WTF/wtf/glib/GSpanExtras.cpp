/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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
#include "config.h"
#include <wtf/glib/GSpanExtras.h>

#include <wtf/StdLibExtras.h>

namespace WTF {

GMallocSpan<char> gFileGetContents(const char* path, GUniqueOutPtr<GError>& error)
{
    char* contents;
    gsize length;
    if (!g_file_get_contents(path, &contents, &length, &error.outPtr()))
        return { };

    return adoptGMallocSpan(unsafeMakeSpan(contents, length));
}

GMallocSpan<char*, GMallocStrv> gKeyFileGetKeys(GKeyFile* keyFile, const char* groupName, GUniqueOutPtr<GError>& error)
{
    ASSERT(keyFile);
    ASSERT(groupName);

    size_t keyCount = 0;
    char** keys = g_key_file_get_keys(keyFile, groupName, &keyCount, &error.outPtr());
    return adoptGMallocSpan<char*, GMallocStrv>(unsafeMakeSpan(keys, keyCount));
}

GMallocSpan<GParamSpec*> gObjectClassGetProperties(GObjectClass* objectClass)
{
    ASSERT(objectClass);

    unsigned propertyCount = 0;
    GParamSpec** properties = g_object_class_list_properties(objectClass, &propertyCount);
    return adoptGMallocSpan(unsafeMakeSpan(properties, propertyCount));
}

GMallocSpan<const char*> gVariantGetStrv(const GRefPtr<GVariant>& variant)
{
    ASSERT(variant);

    size_t stringCount = 0;
    const char** strings = g_variant_get_strv(variant.get(), &stringCount);
    return adoptGMallocSpan(unsafeMakeSpan(strings, stringCount));
}

} // namespace WTF
