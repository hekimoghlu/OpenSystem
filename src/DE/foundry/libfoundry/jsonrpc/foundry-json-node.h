/* foundry-json-node.h
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include <json-glib/json-glib.h>

#include "foundry-util.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#ifndef __GI_SCANNER__

#define _FOUNDRY_JSON_NODE_MAGIC(s) ("@!^%" s)
#define _FOUNDRY_JSON_NODE_MAGIC_C(a,b,c,d) {'@','!','^','%',a,b,c,d}

FOUNDRY_ALIGNED_BEGIN(8)
typedef struct
{
  char bytes[8];
} FoundryJsonNodeMagic
FOUNDRY_ALIGNED_END(8);

typedef struct
{
  FoundryJsonNodeMagic magic;
  const char *val;
} FoundryJsonNodePutString;

typedef struct
{
  FoundryJsonNodeMagic magic;
  double val;
} FoundryJsonNodePutDouble;

typedef struct
{
  FoundryJsonNodeMagic magic;
  gint64 val;
} FoundryJsonNodePutInt;

typedef struct
{
  FoundryJsonNodeMagic magic;
  gboolean val;
} FoundryJsonNodePutBoolean;

typedef struct
{
  FoundryJsonNodeMagic magic;
  const char * const *val;
} FoundryJsonNodePutStrv;

typedef struct
{
  FoundryJsonNodeMagic magic;
  JsonNode *val;
} FoundryJsonNodePutNode;

#define _FOUNDRY_JSON_NODE_PUT_STRING_MAGIC    _FOUNDRY_JSON_NODE_MAGIC("PUTS")
#define _FOUNDRY_JSON_NODE_PUT_STRING_MAGIC_C  _FOUNDRY_JSON_NODE_MAGIC_C('P','U','T','S')
#define FOUNDRY_JSON_NODE_PUT_STRING(_val) \
  (&((FoundryJsonNodePutString) { .magic = {_FOUNDRY_JSON_NODE_PUT_STRING_MAGIC_C}, .val = _val }))

#define _FOUNDRY_JSON_NODE_PUT_STRV_MAGIC      _FOUNDRY_JSON_NODE_MAGIC("PUTZ")
#define _FOUNDRY_JSON_NODE_PUT_STRV_MAGIC_C    _FOUNDRY_JSON_NODE_MAGIC_C('P','U','T','Z')
#define FOUNDRY_JSON_NODE_PUT_STRV(_val) \
  (&((FoundryJsonNodePutStrv) { .magic = {_FOUNDRY_JSON_NODE_PUT_STRV_MAGIC_C}, .val = _val }))

#define _FOUNDRY_JSON_NODE_PUT_INT_MAGIC     _FOUNDRY_JSON_NODE_MAGIC("PUTX")
#define _FOUNDRY_JSON_NODE_PUT_INT_MAGIC_C   _FOUNDRY_JSON_NODE_MAGIC_C('P','U','T','X')
#define FOUNDRY_JSON_NODE_PUT_INT(_val) \
  (&((FoundryJsonNodePutInt) { .magic = {_FOUNDRY_JSON_NODE_PUT_INT_MAGIC_C}, .val = _val }))

#define _FOUNDRY_JSON_NODE_PUT_DOUBLE_MAGIC    _FOUNDRY_JSON_NODE_MAGIC("PUTD")
#define _FOUNDRY_JSON_NODE_PUT_DOUBLE_MAGIC_C  _FOUNDRY_JSON_NODE_MAGIC_C('P','U','T','D')
#define FOUNDRY_JSON_NODE_PUT_DOUBLE(_val) \
  (&((FoundryJsonNodePutDouble) { .magic = {_FOUNDRY_JSON_NODE_PUT_DOUBLE_MAGIC_C}, .val = _val }))

#define _FOUNDRY_JSON_NODE_PUT_BOOLEAN_MAGIC   _FOUNDRY_JSON_NODE_MAGIC("PUTB")
#define _FOUNDRY_JSON_NODE_PUT_BOOLEAN_MAGIC_C _FOUNDRY_JSON_NODE_MAGIC_C('P','U','T','B')
#define FOUNDRY_JSON_NODE_PUT_BOOLEAN(_val) \
  (&((FoundryJsonNodePutBoolean) { .magic = {_FOUNDRY_JSON_NODE_PUT_BOOLEAN_MAGIC_C}, .val = _val }))

#define _FOUNDRY_JSON_NODE_PUT_NODE_MAGIC   _FOUNDRY_JSON_NODE_MAGIC("PUTN")
#define _FOUNDRY_JSON_NODE_PUT_NODE_MAGIC_C _FOUNDRY_JSON_NODE_MAGIC_C('P','U','T','N')
#define FOUNDRY_JSON_NODE_PUT_NODE(_val) \
  (&((FoundryJsonNodePutNode) { .magic = {_FOUNDRY_JSON_NODE_PUT_NODE_MAGIC_C}, .val = _val }))

#define FOUNDRY_JSON_OBJECT_NEW(...) \
  _foundry_json_node_new(NULL, "{", __VA_ARGS__, "}", NULL)

#define FOUNDRY_JSON_ARRAY_NEW(...) \
  _foundry_json_node_new(NULL, "[", __VA_ARGS__, "]", NULL)

_FOUNDRY_EXTERN
JsonNode *_foundry_json_node_new   (gpointer unused,
                                    ...) G_GNUC_NULL_TERMINATED G_GNUC_WARN_UNUSED_RESULT;

typedef struct
{
  FoundryJsonNodeMagic magic;
  const char **valptr;
} FoundryJsonNodeGetString;

typedef struct
{
  FoundryJsonNodeMagic magic;
  char ***valptr;
} FoundryJsonNodeGetStrv;

typedef struct
{
  FoundryJsonNodeMagic magic;
  gint64 *valptr;
} FoundryJsonNodeGetInt;

typedef struct
{
  FoundryJsonNodeMagic magic;
  gboolean *valptr;
} FoundryJsonNodeGetBoolean;

typedef struct
{
  FoundryJsonNodeMagic magic;
  double *valptr;
} FoundryJsonNodeGetDouble;

typedef struct
{
  FoundryJsonNodeMagic magic;
  JsonNode **valptr;
} FoundryJsonNodeGetNode;

#define _FOUNDRY_JSON_NODE_GET_STRING_MAGIC    _FOUNDRY_JSON_NODE_MAGIC("GETS")
#define _FOUNDRY_JSON_NODE_GET_STRING_MAGIC_C  _FOUNDRY_JSON_NODE_MAGIC_C('G','E','T','S')
#define FOUNDRY_JSON_NODE_GET_STRING(_valptr) \
  (&((FoundryJsonNodeGetString) { .magic = {_FOUNDRY_JSON_NODE_GET_STRING_MAGIC_C}, .valptr = _valptr }))

#define _FOUNDRY_JSON_NODE_GET_STRV_MAGIC      _FOUNDRY_JSON_NODE_MAGIC("GETZ")
#define _FOUNDRY_JSON_NODE_GET_STRV_MAGIC_C    _FOUNDRY_JSON_NODE_MAGIC_C('G','E','T','Z')
#define FOUNDRY_JSON_NODE_GET_STRV(_valptr) \
  (&((FoundryJsonNodeGetStrv) { .magic = {_FOUNDRY_JSON_NODE_GET_STRV_MAGIC_C}, .valptr = _valptr }))

#define _FOUNDRY_JSON_NODE_GET_INT_MAGIC     _FOUNDRY_JSON_NODE_MAGIC("GETX")
#define _FOUNDRY_JSON_NODE_GET_INT_MAGIC_C   _FOUNDRY_JSON_NODE_MAGIC_C('G','E','T','X')
#define FOUNDRY_JSON_NODE_GET_INT(_valptr) \
  (&((FoundryJsonNodeGetInt) { .magic = {_FOUNDRY_JSON_NODE_GET_INT_MAGIC_C}, .valptr = _valptr }))

#define _FOUNDRY_JSON_NODE_GET_DOUBLE_MAGIC    _FOUNDRY_JSON_NODE_MAGIC("GETD")
#define _FOUNDRY_JSON_NODE_GET_DOUBLE_MAGIC_C  _FOUNDRY_JSON_NODE_MAGIC_C('G','E','T','D')
#define FOUNDRY_JSON_NODE_GET_DOUBLE(_valptr) \
  (&((FoundryJsonNodeGetDouble) { .magic = {_FOUNDRY_JSON_NODE_GET_DOUBLE_MAGIC_C}, .valptr = _valptr }))

#define _FOUNDRY_JSON_NODE_GET_BOOLEAN_MAGIC   _FOUNDRY_JSON_NODE_MAGIC("GETB")
#define _FOUNDRY_JSON_NODE_GET_BOOLEAN_MAGIC_C _FOUNDRY_JSON_NODE_MAGIC_C('G','E','T','B')
#define FOUNDRY_JSON_NODE_GET_BOOLEAN(_valptr) \
  (&((FoundryJsonNodeGetBoolean) { .magic = {_FOUNDRY_JSON_NODE_GET_BOOLEAN_MAGIC_C}, .valptr = _valptr }))

#define _FOUNDRY_JSON_NODE_GET_NODE_MAGIC   _FOUNDRY_JSON_NODE_MAGIC("GETN")
#define _FOUNDRY_JSON_NODE_GET_NODE_MAGIC_C _FOUNDRY_JSON_NODE_MAGIC_C('G','E','T','N')
#define FOUNDRY_JSON_NODE_GET_NODE(_valptr) \
  (&((FoundryJsonNodeGetNode) { .magic = {_FOUNDRY_JSON_NODE_GET_NODE_MAGIC_C}, .valptr = _valptr }))

#define FOUNDRY_JSON_OBJECT_PARSE(node, ...) \
  _foundry_json_node_parse(node, "{", __VA_ARGS__, "}", NULL)

#define FOUNDRY_JSON_ARRAY_PARSE(node, ...) \
  _foundry_json_node_parse(node, "[", __VA_ARGS__, "]", NULL)

_FOUNDRY_EXTERN
gboolean  _foundry_json_node_parse (JsonNode *node,
                                    ...) G_GNUC_NULL_TERMINATED;

#endif

G_END_DECLS
