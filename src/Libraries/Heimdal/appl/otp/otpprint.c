/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
#include "otp_locl.h"
#include <getarg.h>

RCSID("$Id$");

static int extendedp;
static int count = 10;
static int hexp;
static char* alg_string;
static int version_flag;
static int help_flag;

struct getargs args[] = {
    { "extended", 'e', arg_flag, &extendedp, "print keys in extended format" },
    { "count", 'n', arg_integer, &count, "number of keys to print" },
    { "hexadecimal", 'h', arg_flag, &hexp, "output in hexadecimal" },
    { "hash", 'f', arg_string, &alg_string,
      "hash algorithm (md4, md5, or sha)", "algorithm"},
    { "version", 0, arg_flag, &version_flag },
    { "help", 0, arg_flag, &help_flag }
};

int num_args = sizeof(args) / sizeof(args[0]);

static void
usage(int code)
{
    arg_printusage(args, num_args, NULL, "num seed");
    exit(code);
}

static int
print (int argc,
       char **argv,
       int count,
       OtpAlgorithm *alg,
       void (*print_fn)(OtpKey, char *, size_t))
{
  char pw[64];
  OtpKey key;
  int n;
  int i;
  char *seed;

  if (argc != 2)
      usage (1);
  n = atoi(argv[0]);
  seed = argv[1];
  if (UI_UTIL_read_pw_string (pw, sizeof(pw), "Pass-phrase: ", 0))
    return 1;
  alg->init (key, pw, seed);
  for (i = 0; i < n; ++i) {
    char s[64];

    alg->next (key);
    if (i >= n - count) {
      (*print_fn)(key, s, sizeof(s));
      printf ("%d: %s\n", i + 1, s);
    }
  }
  return 0;
}

int
main (int argc, char **argv)
{
    int optind = 0;
    void (*fn)(OtpKey, char *, size_t);
    OtpAlgorithm *alg = otp_find_alg (OTP_ALG_DEFAULT);

    setprogname (argv[0]);
    if(getarg(args, num_args, argc, argv, &optind))
	usage(1);
    if(help_flag)
	usage(0);
    if(version_flag) {
	print_version(NULL);
	exit(0);
    }

    if(alg_string) {
	alg = otp_find_alg (alg_string);
	if (alg == NULL)
	    errx(1, "Unknown algorithm: %s", alg_string);
    }
    argc -= optind;
    argv += optind;

    if (hexp) {
	if (extendedp)
	    fn = otp_print_hex_extended;
	else
	    fn = otp_print_hex;
    } else {
	if (extendedp)
	    fn = otp_print_stddict_extended;
	else
	    fn = otp_print_stddict;
    }

    return print (argc, argv, count, alg, fn);
}
