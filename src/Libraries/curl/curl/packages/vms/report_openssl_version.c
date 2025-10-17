/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
#include <dlfcn.h>
#include <openssl/opensslv.h>
#include <openssl/crypto.h>

#include <string.h>
#include <descrip.h>
#include <libclidef.h>
#include <stsdef.h>
#include <errno.h>

unsigned long LIB$SET_SYMBOL(
    const struct dsc$descriptor_s * symbol,
    const struct dsc$descriptor_s * value,
    const unsigned long * table_type);

int main(int argc, char ** argv) {


void * libptr;
const char * (*ssl_version)(int t);
const char * version;

   if(argc < 1) {
       puts("report_openssl_version filename");
       exit(1);
   }

   libptr = dlopen(argv[1], 0);

   ssl_version = (const char * (*)(int))dlsym(libptr, "SSLeay_version");
   if(ssl_version == NULL) {
      ssl_version = (const char * (*)(int))dlsym(libptr, "ssleay_version");
      if(ssl_version == NULL) {
         ssl_version = (const char * (*)(int))dlsym(libptr, "SSLEAY_VERSION");
      }
   }

   dlclose(libptr);

   if(ssl_version == NULL) {
      puts("Unable to lookup version of OpenSSL");
      exit(1);
   }

   version = ssl_version(SSLEAY_VERSION);

   puts(version);

   /* Was a symbol argument given? */
   if(argc > 1) {
      int status;
      struct dsc$descriptor_s symbol_dsc;
      struct dsc$descriptor_s value_dsc;
      const unsigned long table_type = LIB$K_CLI_LOCAL_SYM;

      symbol_dsc.dsc$a_pointer = argv[2];
      symbol_dsc.dsc$w_length = strlen(argv[2]);
      symbol_dsc.dsc$b_dtype = DSC$K_DTYPE_T;
      symbol_dsc.dsc$b_class = DSC$K_CLASS_S;

      value_dsc.dsc$a_pointer = (char *)version; /* Cast ok */
      value_dsc.dsc$w_length = strlen(version);
      value_dsc.dsc$b_dtype = DSC$K_DTYPE_T;
      value_dsc.dsc$b_class = DSC$K_CLASS_S;

      status = LIB$SET_SYMBOL(&symbol_dsc, &value_dsc, &table_type);
      if(!$VMS_STATUS_SUCCESS(status)) {
         exit(status);
      }
   }

   exit(0);
}
