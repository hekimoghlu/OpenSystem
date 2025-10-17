/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#include "curlcheck.h"

#include "tool_cfgable.h"
#include "tool_doswin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "memdebug.h" /* LAST include file */

static CURLcode unit_setup(void)
{
  return CURLE_OK;
}

static void unit_stop(void)
{

}

#if defined(_WIN32) || defined(MSDOS)

static char *getflagstr(int flags)
{
  char *buf = malloc(256);
  if(buf) {
    msnprintf(buf, 256, "%s,%s,%s,%s",
              ((flags & SANITIZE_ALLOW_COLONS) ?
               "SANITIZE_ALLOW_COLONS" : ""),
              ((flags & SANITIZE_ALLOW_PATH) ?
               "SANITIZE_ALLOW_PATH" : ""),
              ((flags & SANITIZE_ALLOW_RESERVED) ?
               "SANITIZE_ALLOW_RESERVED" : ""),
              ((flags & SANITIZE_ALLOW_TRUNCATE) ?
               "SANITIZE_ALLOW_TRUNCATE" : ""));
  }
  return buf;
}

static char *getcurlcodestr(int cc)
{
  char *buf = malloc(256);
  if(buf) {
    msnprintf(buf, 256, "%s (%d)",
              (cc == SANITIZE_ERR_OK ? "SANITIZE_ERR_OK" :
               cc == SANITIZE_ERR_BAD_ARGUMENT ? "SANITIZE_ERR_BAD_ARGUMENT" :
               cc == SANITIZE_ERR_INVALID_PATH ? "SANITIZE_ERR_INVALID_PATH" :
               cc == SANITIZE_ERR_OUT_OF_MEMORY ? "SANITIZE_ERR_OUT_OF_MEMORY":
               "unexpected error code - add name"),
              cc);
  }
  return buf;
}

struct data {
  const char *input;
  int flags;
  const char *expected_output;
  SANITIZEcode expected_result;
};

UNITTEST_START
{ /* START sanitize_file_name */
  struct data data[] = {
    { "", 0,
      "", SANITIZE_ERR_OK
    },
    { "normal filename", 0,
      "normal filename", SANITIZE_ERR_OK
    },
    { "control\tchar", 0,
      "control_char", SANITIZE_ERR_OK
    },
    { "banned*char", 0,
      "banned_char", SANITIZE_ERR_OK
    },
    { "f:foo", 0,
      "f_foo", SANITIZE_ERR_OK
    },
    { "f:foo", SANITIZE_ALLOW_COLONS,
      "f:foo", SANITIZE_ERR_OK
    },
    { "f:foo", SANITIZE_ALLOW_PATH,
      "f:foo", SANITIZE_ERR_OK
    },
    { "f:\\foo", 0,
      "f__foo", SANITIZE_ERR_OK
    },
    { "f:\\foo", SANITIZE_ALLOW_PATH,
      "f:\\foo", SANITIZE_ERR_OK
    },
    { "f:/foo", 0,
      "f__foo", SANITIZE_ERR_OK
    },
    { "f:/foo", SANITIZE_ALLOW_PATH,
      "f:/foo", SANITIZE_ERR_OK
    },
#ifndef MSDOS
    { "\\\\?\\C:\\foo", SANITIZE_ALLOW_PATH,
      "\\\\?\\C:\\foo", SANITIZE_ERR_OK
    },
    { "\\\\?\\C:\\foo", 0,
      "____C__foo", SANITIZE_ERR_OK
    },
#endif
    { "foo:bar", 0,
      "foo_bar", SANITIZE_ERR_OK
    },
    { "foo|<>/bar\\\":?*baz", 0,
      "foo____bar_____baz", SANITIZE_ERR_OK
    },
    { "f:foo::$DATA", 0,
      "f_foo__$DATA", SANITIZE_ERR_OK
    },
    { "con . air", 0,
      "con _ air", SANITIZE_ERR_OK
    },
    { "con.air", 0,
      "con_air", SANITIZE_ERR_OK
    },
    { "con:/x", 0,
      "con__x", SANITIZE_ERR_OK
    },
    { "file . . . .  ..  .", 0,
      "file", SANITIZE_ERR_OK
    },
    { "foo . . ? . . ", 0,
      "foo . . _", SANITIZE_ERR_OK
    },
    { "com1", 0,
      "_com1", SANITIZE_ERR_OK
    },
    { "com1", SANITIZE_ALLOW_RESERVED,
      "com1", SANITIZE_ERR_OK
    },
    { "f:\\com1", 0,
      "f__com1", SANITIZE_ERR_OK
    },
    { "f:\\com1", SANITIZE_ALLOW_PATH,
      "f:\\_com1", SANITIZE_ERR_OK
    },
    { "f:\\com1", SANITIZE_ALLOW_RESERVED,
      "f__com1", SANITIZE_ERR_OK
    },
    { "f:\\com1", SANITIZE_ALLOW_RESERVED | SANITIZE_ALLOW_COLONS,
      "f:_com1", SANITIZE_ERR_OK
    },
    { "f:\\com1", SANITIZE_ALLOW_RESERVED | SANITIZE_ALLOW_PATH,
      "f:\\com1", SANITIZE_ERR_OK
    },
    { "com1:\\com1", SANITIZE_ALLOW_PATH,
      "_com1:\\_com1", SANITIZE_ERR_OK
    },
    { "com1:\\com1", SANITIZE_ALLOW_RESERVED | SANITIZE_ALLOW_PATH,
      "com1:\\com1", SANITIZE_ERR_OK
    },
    { "com1:\\com1", SANITIZE_ALLOW_RESERVED,
      "com1__com1", SANITIZE_ERR_OK
    },
#ifndef MSDOS
    { "\\com1", SANITIZE_ALLOW_PATH,
      "\\_com1", SANITIZE_ERR_OK
    },
    { "\\\\com1", SANITIZE_ALLOW_PATH,
      "\\\\com1", SANITIZE_ERR_OK
    },
    { "\\\\?\\C:\\com1", SANITIZE_ALLOW_PATH,
      "\\\\?\\C:\\com1", SANITIZE_ERR_OK
    },
#endif
    { "CoM1", 0,
      "_CoM1", SANITIZE_ERR_OK
    },
    { "CoM1", SANITIZE_ALLOW_RESERVED,
      "CoM1", SANITIZE_ERR_OK
    },
    { "COM56", 0,
      "COM56", SANITIZE_ERR_OK
    },
    /* At the moment we expect a maximum path length of 259. I assume MSDOS
       has variable max path lengths depending on compiler that are shorter
       so currently these "good" truncate tests won't run on MSDOS */
#ifndef MSDOS
    { "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
      "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
      "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
      "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
      "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        SANITIZE_ALLOW_TRUNCATE,
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
      "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
      "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
      "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
      "FFFFF", SANITIZE_ERR_OK
    },
    { "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
      "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
      "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
      "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
      "FFF\\FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        SANITIZE_ALLOW_TRUNCATE | SANITIZE_ALLOW_PATH,
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
      "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
      "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
      "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
      "FFF\\FFFFF", SANITIZE_ERR_OK
    },
    { "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
      "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
      "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
      "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
      "FFF\\FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        SANITIZE_ALLOW_TRUNCATE,
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
      "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
      "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
      "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
      "FFF_F", SANITIZE_ERR_OK
    },
#endif /* !MSDOS */
    { "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
      "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
      "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
      "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
      "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        0,
      NULL, SANITIZE_ERR_INVALID_PATH
    },
    { "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
      "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
      "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
      "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
      "FFFF\\FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        SANITIZE_ALLOW_TRUNCATE,
      NULL, SANITIZE_ERR_INVALID_PATH
    },
    { "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
      "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
      "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
      "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
      "FFFFFFFFFFFFFFFFFFFFFFFFF\\FFFFFFFFFFFFFFFFFFFFFFFF",
        SANITIZE_ALLOW_TRUNCATE | SANITIZE_ALLOW_PATH,
      NULL, SANITIZE_ERR_INVALID_PATH
    },
    { "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
      "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
      "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
      "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
      "FFF\\FFFFFFFFFFFFFFFFFFFFF:FFFFFFFFFFFFFFFFFFFFFFFF",
        SANITIZE_ALLOW_TRUNCATE | SANITIZE_ALLOW_PATH,
      NULL, SANITIZE_ERR_INVALID_PATH
    },
    { "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
      "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
      "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
      "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
      "FF\\F:FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        SANITIZE_ALLOW_TRUNCATE | SANITIZE_ALLOW_PATH,
      NULL, SANITIZE_ERR_INVALID_PATH
    },
    { NULL, 0,
      NULL, SANITIZE_ERR_BAD_ARGUMENT
    },
  };

  size_t i;

  for(i = 0; i < sizeof(data) / sizeof(data[0]); ++i) {
    char *output = NULL;
    char *flagstr = NULL;
    char *received_ccstr = NULL;
    char *expected_ccstr = NULL;
    SANITIZEcode res;

    res = sanitize_file_name(&output, data[i].input, data[i].flags);

    if(res == data[i].expected_result &&
       ((!output && !data[i].expected_output) ||
        (output && data[i].expected_output &&
         !strcmp(output, data[i].expected_output)))) { /* OK */
      free(output);
      continue;
    }

    flagstr = getflagstr(data[i].flags);
    abort_unless(flagstr, "out of memory");
    received_ccstr = getcurlcodestr(res);
    abort_unless(received_ccstr, "out of memory");
    expected_ccstr = getcurlcodestr(data[i].expected_result);
    abort_unless(expected_ccstr, "out of memory");

    unitfail++;
    fprintf(stderr, "\n"
            "%s:%d sanitize_file_name failed.\n"
            "input: %s\n"
            "flags: %s\n"
            "output: %s\n"
            "result: %s\n"
            "expected output: %s\n"
            "expected result: %s\n",
            __FILE__, __LINE__,
            data[i].input,
            flagstr,
            (output ? output : "(null)"),
            received_ccstr,
            (data[i].expected_output ? data[i].expected_output : "(null)"),
            expected_ccstr);

    free(output);
    free(flagstr);
    free(received_ccstr);
    free(expected_ccstr);
  }
} /* END sanitize_file_name */

#else
UNITTEST_START
{
  fprintf(stderr, "Skipped test not for this platform\n");
}
#endif /* _WIN32 || MSDOS */

UNITTEST_STOP
