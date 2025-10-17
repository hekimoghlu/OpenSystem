/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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
#include <stdio.h>
#include <expat.h>

#ifdef XML_LARGE_SIZE
#  define XML_FMT_INT_MOD "ll"
#else
#  define XML_FMT_INT_MOD "l"
#endif

#ifdef XML_UNICODE_WCHAR_T
#  define XML_FMT_STR "ls"
#else
#  define XML_FMT_STR "s"
#endif

static void XMLCALL
startElement(void *userData, const XML_Char *name, const XML_Char **atts) {
  int i;
  int *const depthPtr = (int *)userData;

  for (i = 0; i < *depthPtr; i++)
    printf("  ");

  printf("%" XML_FMT_STR, name);

  for (i = 0; atts[i]; i += 2) {
    printf(" %" XML_FMT_STR "='%" XML_FMT_STR "'", atts[i], atts[i + 1]);
  }

  printf("\n");
  *depthPtr += 1;
}

static void XMLCALL
endElement(void *userData, const XML_Char *name) {
  int *const depthPtr = (int *)userData;
  (void)name;

  *depthPtr -= 1;
}

int
main(void) {
  XML_Parser parser = XML_ParserCreate(NULL);
  int done;
  int depth = 0;

  if (! parser) {
    fprintf(stderr, "Couldn't allocate memory for parser\n");
    return 1;
  }

  XML_SetUserData(parser, &depth);
  XML_SetElementHandler(parser, startElement, endElement);

  do {
    void *const buf = XML_GetBuffer(parser, BUFSIZ);
    if (! buf) {
      fprintf(stderr, "Couldn't allocate memory for buffer\n");
      XML_ParserFree(parser);
      return 1;
    }

    const size_t len = fread(buf, 1, BUFSIZ, stdin);

    if (ferror(stdin)) {
      fprintf(stderr, "Read error\n");
      XML_ParserFree(parser);
      return 1;
    }

    done = feof(stdin);

    if (XML_ParseBuffer(parser, (int)len, done) == XML_STATUS_ERROR) {
      fprintf(stderr,
              "Parse error at line %" XML_FMT_INT_MOD "u:\n%" XML_FMT_STR "\n",
              XML_GetCurrentLineNumber(parser),
              XML_ErrorString(XML_GetErrorCode(parser)));
      XML_ParserFree(parser);
      return 1;
    }
  } while (! done);

  XML_ParserFree(parser);
  return 0;
}

