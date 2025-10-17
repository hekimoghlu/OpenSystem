/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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
#ifndef json_support_h
#define json_support_h

/*
 Example output

 ./getattrlist -b -f JSON /Volumes/Data/SMBBasic/AppleTest CMN_NAME,ATTR_CMN_OBJTYPE
 {
   "inputs" : {
     "api" : "getattrlistbulk",
     "attributes" : "ATTR_CMN_RETURNED_ATTRS,ATTR_CMN_NAME,ATTR_CMN_OBJTYPE",
     "options" : "<none>",
     "path" : "\/Volumes\/Data\/SMBBasic\/AppleTest",
     "recurse" : "no"
   },
   "outputs" : {
     "\/Volumes\/Data\/SMBBasic\/AppleTest" : {
       "dir_entry_count" : 3,
       "entry-0" : {
         "ATTR_CMN_NAME" : ".DS_Store",
         "ATTR_CMN_OBJTYPE" : "VREG",
         "ATTR_CMN_RETURNED_ATTRS" : "Requested - commonattr 0x80000009 volattr 0x00000000 dirattr 0x00000000 fileattr 0x00000000 forkattr 0x00000000                        Returned  - commonattr 0x80000009 volattr 0x00000000 dirattr 0x00000000 fileattr 0x00000000 forkattr 0x00000000",
         "attr_len" : 48
       },
       "entry-1" : {
         "ATTR_CMN_NAME" : "ShareFolder1",
         "ATTR_CMN_OBJTYPE" : "VDIR",
         "ATTR_CMN_RETURNED_ATTRS" : "Requested - commonattr 0x80000009 volattr 0x00000000 dirattr 0x00000000 fileattr 0x00000000 forkattr 0x00000000                        Returned  - commonattr 0x80000009 volattr 0x00000000 dirattr 0x00000000 fileattr 0x00000000 forkattr 0x00000000",
         "attr_len" : 56
       },
       "entry-2" : {
         "ATTR_CMN_NAME" : "ShareFolder2",
         "ATTR_CMN_OBJTYPE" : "VDIR",
         "ATTR_CMN_RETURNED_ATTRS" : "Requested - commonattr 0x80000009 volattr 0x00000000 dirattr 0x00000000 fileattr 0x00000000 forkattr 0x00000000                        Returned  - commonattr 0x80000009 volattr 0x00000000 dirattr 0x00000000 fileattr 0x00000000 forkattr 0x00000000",
         "attr_len" : 56
       },
       "timings" : {
         "duration_usec" : 615
       }
     }
   },
   "results" : {
     "error" : 0
   }
 }

 */


#pragma mark -

/*
* Add object to an existing dictionary functions
*/

int json_add_array(CFMutableDictionaryRef dict, const char *key,
                   const char *comma_sep_string);
int json_add_cfstr(CFMutableDictionaryRef dict, const char *key,
                    const CFMutableStringRef value);
int json_add_dict(CFMutableDictionaryRef dict, const char *key,
                   const CFMutableDictionaryRef value);
int json_add_bool(CFMutableDictionaryRef dict, const char *key,
                  bool value);
int json_add_num(CFMutableDictionaryRef dict, const char *key,
                  const void *value, size_t size);
int json_add_str(CFMutableDictionaryRef dict, const char *key,
                  const char *value);


#pragma mark -

/*
* Special purpose dictionaries inside of a parent dictionary
* If the special dictionary does not already exist inside the parent
* dictionary, then create it and add it into the parent dictionary.
*/
int json_add_inputs_str(CFMutableDictionaryRef dict, const char *key,
                        const void *value);
int json_add_outputs_dict(CFMutableDictionaryRef dict, const char *key,
                          const CFMutableDictionaryRef value);
int json_add_outputs_str(CFMutableDictionaryRef dict, const char *key,
                         const void *value);
int json_add_results(CFMutableDictionaryRef dict, const char *key,
                     const void *value, size_t size);
int json_add_results_str(CFMutableDictionaryRef dict, const char *key,
                         const void *value);
int json_add_time_stamp(CFMutableDictionaryRef dict, const char *key);
int json_add_timing(CFMutableDictionaryRef dict, const char *key,
                     const void *value, size_t size);


#pragma mark -

/*
* Print out a Core Foundation object in JSON format
*/

int json_print_cf_object(CFTypeRef cf_object, char *output_file_path);

#endif /* json_support_h */
