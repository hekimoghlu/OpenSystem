/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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


// Have to be careful regarding the alignment of structures since
// this file is shared between 32 bit kernel code and 32 or 64 bit
// user level code. 
// See http://developer.apple.com/documentation/Darwin/Conceptual/64bitPorting/index.html

// Structures in this file are aligned naturally by ordering any 64 bit quantities first.
// #pragma pack could also be used to force alignment

enum {
    kAppleSampleMethod1 = 0,
    kAppleSampleMethod2 = 1,
    kAppleSampleMethod3 = 2,
    kAppleSampleNumMethods
};

struct AppleSampleStructForMethod2 {
    mach_vm_address_t   data_pointer;
    mach_vm_size_t      data_length;
    uint32_t            parameter1;
    uint32_t            __pad;
};
#ifndef __cplusplus
typedef struct AppleSampleStructForMethod2 AppleSampleStructForMethod2;
#endif

struct AppleSampleResultsForMethod2 {
    uint64_t            results1;
};
#ifndef __cplusplus
typedef struct AppleSampleResultsForMethod2 AppleSampleResultsForMethod2;
#endif


#define kAppleSamplePCIClassName        "AppleSamplePCI"

// types for IOServiceOpen()
enum {
    kAppleSamplePCIConnectType = 23
};

// types for IOConnectMapMemory()
enum {
    kAppleSamplePCIMemoryType1 = 100,
    kAppleSamplePCIMemoryType2 = 101,
};

// memory structure to be shared between the kernel user client object and client.
struct AppleSampleSharedMemory {
    uint32_t    field1;
    uint32_t    field2;
    uint32_t    field3;
    char        string[100];
};
#ifndef __cplusplus
typedef struct AppleSampleSharedMemory AppleSampleSharedMemory;
#endif

