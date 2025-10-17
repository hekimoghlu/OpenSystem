/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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
#ifndef ExternalGenericMetadataBuilderImport
#define ExternalGenericMetadataBuilderImport

#include <stdint.h>

#define WEAK_IMPORT_ATTR __attribute__((weak_import))

#ifdef __cplusplus
extern "C" {
#endif

struct SwiftExternalMetadataBuilder;
struct mach_header;

// Create a builder object with the given platform and architecture name.
WEAK_IMPORT_ATTR
struct SwiftExternalMetadataBuilder *
swift_externalMetadataBuilder_create(int platform, const char *arch);

// Destroy a builder object.
WEAK_IMPORT_ATTR
void swift_externalMetadataBuilder_destroy(
    struct SwiftExternalMetadataBuilder *);

// Returns an error string if the dylib could not be added
// The builder owns the string, so the caller does not have to free it
// The mach_header* is the raw dylib from disk/memory, before the shared cache
// builder has created its own copy of it
WEAK_IMPORT_ATTR
const char *swift_externalMetadataBuilder_addDylib(
    struct SwiftExternalMetadataBuilder *, const char *install_name,
    const struct mach_header *, uint64_t size);

WEAK_IMPORT_ATTR
const char *swift_externalMetadataBuilder_readNamesJSON(
    struct SwiftExternalMetadataBuilder *, const char *names_json);

// Returns an error string if the dylib could not be added
// The builder owns the string, so the caller does not have to free it
WEAK_IMPORT_ATTR
const char *swift_externalMetadataBuilder_buildMetadata(
    struct SwiftExternalMetadataBuilder *);

// Get the JSON for the built metadata
WEAK_IMPORT_ATTR
const char *swift_externalMetadataBuilder_getMetadataJSON(
    struct SwiftExternalMetadataBuilder *);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ExternalGenericMetadataBuilderImport
