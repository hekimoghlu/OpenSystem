/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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
#ifdef __cplusplus
extern "C" {
#endif

//
// Reads from (possibly fat) mach-o file at path 'existingFilePath'
// and writes a new file at path 'newOutputFilePath'.
// Returns true on success, otherwise returns false and the errorMessage
// parameter is set to a malloced failure messsage.
// For each mach-o slice in the file, the block 'getSigSpaceNeeded' is called
// and expected to return the amount of space need for the code signature.
//
extern bool code_sign_allocate(const char* existingFilePath,
                               const char* newOutputFilePath,
                               unsigned int (^getSigSpaceNeeded)(cpu_type_t cputype, cpu_subtype_t cpusubtype),
                               char*& errorMessage);


//
// Reads from (possibly fat) mach-o file at path 'existingFilePath'
// and writes a new file at path 'newOutputFilePath'.  For each
// mach-o slice in the file, if there is an existing code signature,
// the code signature data in the LINKEDIT is removed along with
// the LC_CODE_SIGNATURE load command.  Returns true on success,
// otherwise returns false and the errorMessage parameter is set
// to a malloced failure messsage.
//
bool code_sign_deallocate(const char* existingFilePath,
                          const char* newOutputFilePath,
                          char*& errorMessage);

#ifdef __cplusplus
}
#endif


