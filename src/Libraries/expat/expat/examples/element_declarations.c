/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
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

// While traversing the XML_Content tree, we avoid recursion
// to not be vulnerable to a denial of service attack.
typedef struct StackStruct {
  const XML_Content *model;
  unsigned level;
  struct StackStruct *prev;
} Stack;

static Stack *
stackPushMalloc(Stack *stackTop, const XML_Content *model, unsigned level) {
  Stack *const newStackTop = malloc(sizeof(Stack));
  if (! newStackTop) {
    return NULL;
  }
  newStackTop->model = model;
  newStackTop->level = level;
  newStackTop->prev = stackTop;
  return newStackTop;
}

static Stack *
stackPopFree(Stack *stackTop) {
  Stack *const newStackTop = stackTop->prev;
  free(stackTop);
  return newStackTop;
}

static char *
contentTypeName(enum XML_Content_Type contentType) {
  switch (contentType) {
  case XML_CTYPE_EMPTY:
    return "EMPTY";
  case XML_CTYPE_ANY:
    return "ANY";
  case XML_CTYPE_MIXED:
    return "MIXED";
  case XML_CTYPE_NAME:
    return "NAME";
  case XML_CTYPE_CHOICE:
    return "CHOICE";
  case XML_CTYPE_SEQ:
    return "SEQ";
  default:
    return "???";
  }
}

static char *
contentQuantName(enum XML_Content_Quant contentQuant) {
  switch (contentQuant) {
  case XML_CQUANT_NONE:
    return "NONE";
  case XML_CQUANT_OPT:
    return "OPT";
  case XML_CQUANT_REP:
    return "REP";
  case XML_CQUANT_PLUS:
    return "PLUS";
  default:
    return "???";
  }
}

static void
dumpContentModelElement(const XML_Content *model, unsigned level,
                        const XML_Content *root) {
  // Indent
  unsigned u = 0;
  for (; u < level; u++) {
    printf("  ");
  }

  // Node
  printf("[%u] type=%s(%u), quant=%s(%u)", (unsigned)(model - root),
         contentTypeName(model->type), (unsigned int)model->type,
         contentQuantName(model->quant), (unsigned int)model->quant);
  if (model->name) {
    printf(", name=\"%" XML_FMT_STR "\"", model->name);
  } else {
    printf(", name=NULL");
  }
  printf(", numchildren=%u", model->numchildren);
  printf("\n");
}

static bool
dumpContentModel(const XML_Char *name, const XML_Content *root) {
  printf("Element \"%" XML_FMT_STR "\":\n", name);
  Stack *stackTop = stackPushMalloc(NULL, root, 1);
  if (! stackTop) {
    return false;
  }

  while (stackTop) {
    const XML_Content *const model = stackTop->model;
    const unsigned level = stackTop->level;

    dumpContentModelElement(model, level, root);

    stackTop = stackPopFree(stackTop);

    for (size_t u = model->numchildren; u >= 1; u--) {
      Stack *const newStackTop
          = stackPushMalloc(stackTop, model->children + (u - 1), level + 1);
      if (! newStackTop) {
        // We ran out of memory, so let's free all memory allocated
        // earlier in this function, to be leak-clean:
        while (stackTop != NULL) {
          stackTop = stackPopFree(stackTop);
        }
        return false;
      }
      stackTop = newStackTop;
    }
  }

  printf("\n");
  return true;
}

static void XMLCALL
handleElementDeclaration(void *userData, const XML_Char *name,
                         XML_Content *model) {
  XML_Parser parser = (XML_Parser)userData;
  const bool success = dumpContentModel(name, model);
  XML_FreeContentModel(parser, model);
  if (! success) {
    XML_StopParser(parser, /* resumable= */ XML_FALSE);
  }
}

int
main(void) {
  XML_Parser parser = XML_ParserCreate(NULL);
  int done;

  if (! parser) {
    fprintf(stderr, "Couldn't allocate memory for parser\n");
    return 1;
  }

  XML_SetUserData(parser, parser);
  XML_SetElementDeclHandler(parser, handleElementDeclaration);

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
      enum XML_Error errorCode = XML_GetErrorCode(parser);
      if (errorCode == XML_ERROR_ABORTED) {
        errorCode = XML_ERROR_NO_MEMORY;
      }
      fprintf(stderr,
              "Parse error at line %" XML_FMT_INT_MOD "u:\n%" XML_FMT_STR "\n",
              XML_GetCurrentLineNumber(parser), XML_ErrorString(errorCode));
      XML_ParserFree(parser);
      return 1;
    }
  } while (! done);

  XML_ParserFree(parser);
  return 0;
}

