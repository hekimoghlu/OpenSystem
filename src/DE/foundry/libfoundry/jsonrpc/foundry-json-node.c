/* foundry-json-node.c
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

#include "config.h"

#include "foundry-json.h"
#include "foundry-json-node.h"

#define COMPARE_MAGIC(_any,_magic) \
  (strncmp ((const char *)_any, \
            _FOUNDRY_JSON_NODE_##_magic##_MAGIC, \
            sizeof (FoundryJsonNodeMagic)) == 0)

#define IS_GET_STRING(_any)  COMPARE_MAGIC(_any, GET_STRING)
#define IS_GET_STRV(_any)    COMPARE_MAGIC(_any, GET_STRV)
#define IS_GET_INT(_any)     COMPARE_MAGIC(_any, GET_INT)
#define IS_GET_BOOLEAN(_any) COMPARE_MAGIC(_any, GET_BOOLEAN)
#define IS_GET_DOUBLE(_any)  COMPARE_MAGIC(_any, GET_DOUBLE)
#define IS_GET_NODE(_any)    COMPARE_MAGIC(_any, GET_NODE)

static JsonNode *
from_string (const char *valueptr)
{
  if (valueptr == NULL)
    return json_node_new (JSON_NODE_NULL);

  {
    JsonNode *node = json_node_new (JSON_NODE_VALUE);
    json_node_set_string (node, valueptr);
    return node;
  }
}

static JsonNode *
from_put_string (FoundryJsonNodePutString *valueptr)
{
  return from_string (valueptr->val);
}

static JsonNode *
from_put_strv (FoundryJsonNodePutStrv *valueptr)
{
  if (valueptr->val == NULL)
    return json_node_new (JSON_NODE_NULL);

  return foundry_json_node_new_strv (valueptr->val);
}

static JsonNode *
from_put_double (FoundryJsonNodePutDouble *valueptr)
{
  JsonNode *node = json_node_new (JSON_NODE_VALUE);
  json_node_set_double (node, valueptr->val);
  return node;
}

static JsonNode *
from_put_int (FoundryJsonNodePutInt *valueptr)
{
  JsonNode *node = json_node_new (JSON_NODE_VALUE);
  json_node_set_int (node, valueptr->val);
  return node;
}

static JsonNode *
from_put_boolean (FoundryJsonNodePutBoolean *valueptr)
{
  JsonNode *node = json_node_new (JSON_NODE_VALUE);
  json_node_set_boolean (node, !!valueptr->val);
  return node;
}

static JsonNode *
from_put_node (FoundryJsonNodePutNode *valueptr)
{
  if (valueptr->val)
    return json_node_ref (valueptr->val);

  return json_node_new (JSON_NODE_NULL);
}

static JsonNode *
create_for_value (va_list *args)
{
  const char *valueptr = va_arg ((*args), const char *);

  if (memcmp (valueptr, "]", 2) == 0)
    return NULL;

  if (memcmp (valueptr, "{", 2) == 0)
    {
      g_autoptr(JsonObject) object = json_object_new ();
      const char *key = va_arg (*args, const char *);
      JsonNode *node;

      node = json_node_new (JSON_NODE_OBJECT);
      json_node_set_object (node, object);

      while (memcmp (key, "}", 2) != 0)
        {
          JsonNode *value = create_for_value (args);
          json_object_set_member (object, key, g_steal_pointer (&value));
          key = va_arg (*args, const char *);
        }

      return node;
    }
  else if (valueptr[0] == '[' && valueptr[1] == 0)
    {
      g_autoptr(JsonArray) array = json_array_new ();
      JsonNode *node;
      JsonNode *element;

      node = json_node_new (JSON_NODE_ARRAY);
      json_node_set_array (node, array);

      while ((element = create_for_value (args)))
        json_array_add_element (array, g_steal_pointer (&element));

      return node;
    }
  else if (strncmp (valueptr, _FOUNDRY_JSON_NODE_PUT_STRING_MAGIC, 8) == 0)
    return from_put_string ((FoundryJsonNodePutString *)(gpointer)valueptr);
  else if (strncmp (valueptr, _FOUNDRY_JSON_NODE_PUT_STRV_MAGIC, 8) == 0)
    return from_put_strv ((FoundryJsonNodePutStrv *)(gpointer)valueptr);
  else if (strncmp (valueptr, _FOUNDRY_JSON_NODE_PUT_DOUBLE_MAGIC, 8) == 0)
    return from_put_double ((FoundryJsonNodePutDouble *)(gpointer)valueptr);
  else if (strncmp (valueptr, _FOUNDRY_JSON_NODE_PUT_BOOLEAN_MAGIC, 8) == 0)
    return from_put_boolean ((FoundryJsonNodePutBoolean *)(gpointer)valueptr);
  else if (strncmp (valueptr, _FOUNDRY_JSON_NODE_PUT_INT_MAGIC, 8) == 0)
    return from_put_int ((FoundryJsonNodePutInt *)(gpointer)valueptr);
  else if (strncmp (valueptr, _FOUNDRY_JSON_NODE_PUT_NODE_MAGIC, 8) == 0)
    return from_put_node ((FoundryJsonNodePutNode *)(gpointer)valueptr);
  else
    return from_string (valueptr);
}

/**
 * _foundry_json_node_new:
 *
 * Returns: (transfer full):
 */
JsonNode *
_foundry_json_node_new (gpointer unused,
                        ...)
{
  JsonNode *ret;
  va_list args;

  g_return_val_if_fail (unused == NULL, NULL);

  va_start (args, unused);
  ret = create_for_value (&args);
  va_end (args);

  return ret;
}

static gboolean
foundry_json_node_parse_array (JsonNode *node,
                               va_list  *args);

static gboolean
foundry_json_node_parse_object (JsonNode *node,
                                va_list  *args)
{
  JsonObject *obj;

  if (node == NULL ||
      !JSON_NODE_HOLDS_OBJECT (node) ||
      !(obj = json_node_get_object (node)))
    return FALSE;

  for (gpointer ptr = va_arg (*args, gpointer);
       memcmp (ptr, "}", 2) != 0;
       ptr = va_arg (*args, gpointer))
    {
      const char *key = ptr;
      gpointer magic;
      JsonNode *member;

      if (!json_object_has_member (obj, key))
        return FALSE;

      member = json_object_get_member (obj, key);
      magic = va_arg (*args, gpointer);

      if (memcmp (magic, "[", 2) == 0)
        {
          if (!foundry_json_node_parse_array (member, args))
            return FALSE;
        }
      else if (memcmp (magic, "{", 2) == 0)
        {
          if (!foundry_json_node_parse_object (member, args))
            return FALSE;
        }
      else if (IS_GET_STRING (magic))
        {
          if (!JSON_NODE_HOLDS_VALUE (member))
            return FALSE;

          if (json_node_get_value_type (member) != G_TYPE_STRING)
            return FALSE;

          *((FoundryJsonNodeGetString *)magic)->valptr = json_node_get_string (member);
        }
      else if (IS_GET_STRV (magic))
        {
          g_autoptr(GStrvBuilder) builder = NULL;
          JsonArray *str_ar;
          guint length;

          if (!JSON_NODE_HOLDS_ARRAY (member))
            return FALSE;

          if (!(str_ar = json_node_get_array (member)))
            return FALSE;

          length = json_array_get_length (str_ar);
          builder = g_strv_builder_new ();

          for (guint j = 0; j < length; j++)
            {
              JsonNode *str_e = json_array_get_element (str_ar, j);

              if (!JSON_NODE_HOLDS_VALUE (str_e) ||
                  json_node_get_value_type (str_e) != G_TYPE_STRING)
                return FALSE;

              g_strv_builder_add (builder, json_node_get_string (str_e));
            }

          *((FoundryJsonNodeGetStrv *)magic)->valptr = g_strv_builder_end (builder);
        }
      else if (IS_GET_INT (magic))
        {
          if (!JSON_NODE_HOLDS_VALUE (member))
            return FALSE;

          if (json_node_get_value_type (member) != G_TYPE_INT64)
            return FALSE;

          *((FoundryJsonNodeGetInt *)magic)->valptr = json_node_get_int (member);
        }
      else if (IS_GET_BOOLEAN (magic))
        {
          if (!JSON_NODE_HOLDS_VALUE (member))
            return FALSE;

          if (json_node_get_value_type (member) != G_TYPE_BOOLEAN)
            return FALSE;

          *((FoundryJsonNodeGetBoolean *)magic)->valptr = json_node_get_boolean (member);
        }
      else if (IS_GET_DOUBLE (magic))
        {
          if (!JSON_NODE_HOLDS_VALUE (member))
            return FALSE;

          if (json_node_get_value_type (member) != G_TYPE_DOUBLE)
            return FALSE;

          *((FoundryJsonNodeGetDouble *)magic)->valptr = json_node_get_double (member);
        }
      else if (IS_GET_NODE (magic))
        {
          *((FoundryJsonNodeGetNode *)magic)->valptr = member;
        }
      else
        {
          if (!JSON_NODE_HOLDS_VALUE (member))
            return FALSE;

          if (json_node_get_value_type (member) != G_TYPE_STRING)
            return FALSE;

          if (g_strcmp0 (magic, json_node_get_string (member)) != 0)
            return FALSE;
        }
    }

  return TRUE;
}

static gboolean
foundry_json_node_parse_array (JsonNode *node,
                               va_list  *args)
{
  JsonArray *ar;
  guint i = 0;
  guint length;

  if (node == NULL ||
      !JSON_NODE_HOLDS_ARRAY (node) ||
      !(ar = json_node_get_array (node)))
    return FALSE;

  length = json_array_get_length (ar);

  for (gpointer ptr = va_arg (*args, gpointer);
       memcmp (ptr, "]", 2) != 0;
       ptr = va_arg (*args, gpointer))
    {
      JsonNode *element;
      gpointer magic = ptr;

      if (i >= length)
        return FALSE;

      element = json_array_get_element (ar, i++);

      if (memcmp (magic, "[", 2) == 0)
        {
          if (!foundry_json_node_parse_array (element, args))
            return FALSE;
        }
      else if (memcmp (magic, "{", 2) == 0)
        {
          if (!foundry_json_node_parse_object (element, args))
            return FALSE;
        }
      else if (IS_GET_STRING (magic))
        {
          if (!JSON_NODE_HOLDS_VALUE (element))
            return FALSE;

          if (json_node_get_value_type (element) != G_TYPE_STRING)
            return FALSE;

          *((FoundryJsonNodeGetString *)magic)->valptr = json_node_get_string (element);
        }
      else if (IS_GET_STRV (magic))
        {
          g_autoptr(GStrvBuilder) builder = NULL;
          JsonArray *str_ar;
          guint str_l;

          if (!JSON_NODE_HOLDS_ARRAY (element))
            return FALSE;

          if (!(str_ar = json_node_get_array (element)))
            return FALSE;

          str_l = json_array_get_length (str_ar);
          builder = g_strv_builder_new ();

          for (guint j = 0; j < str_l; j++)
            {
              JsonNode *str_e = json_array_get_element (str_ar, j);

              if (!JSON_NODE_HOLDS_VALUE (str_e) ||
                  json_node_get_value_type (str_e) != G_TYPE_STRING)
                return FALSE;

              g_strv_builder_add (builder, json_node_get_string (str_e));
            }

          *((FoundryJsonNodeGetStrv *)magic)->valptr = g_strv_builder_end (builder);
        }
      else if (IS_GET_INT (magic))
        {
          if (!JSON_NODE_HOLDS_VALUE (element))
            return FALSE;

          if (json_node_get_value_type (element) != G_TYPE_INT64)
            return FALSE;

          *((FoundryJsonNodeGetInt *)magic)->valptr = json_node_get_int (element);
        }
      else if (IS_GET_BOOLEAN (magic))
        {
          if (!JSON_NODE_HOLDS_VALUE (element))
            return FALSE;

          if (json_node_get_value_type (element) != G_TYPE_BOOLEAN)
            return FALSE;

          *((FoundryJsonNodeGetBoolean *)magic)->valptr = json_node_get_boolean (element);
        }
      else if (IS_GET_DOUBLE (magic))
        {
          if (!JSON_NODE_HOLDS_VALUE (element))
            return FALSE;

          if (json_node_get_value_type (element) != G_TYPE_DOUBLE)
            return FALSE;

          *((FoundryJsonNodeGetDouble *)magic)->valptr = json_node_get_double (element);
        }
      else if (IS_GET_NODE (magic))
        {
          *((FoundryJsonNodeGetNode *)magic)->valptr = element;
        }
      else
        {
          if (!JSON_NODE_HOLDS_VALUE (element))
            return FALSE;

          if (json_node_get_value_type (element) != G_TYPE_STRING)
            return FALSE;

          if (g_strcmp0 (magic, json_node_get_string (element)) != 0)
            return FALSE;
        }
    }

  return i == length;
}

static gboolean
foundry_json_node_parse_valist (JsonNode *node,
                                va_list  *args)
{
  gpointer param;

  g_assert (node != NULL);

  if (!(param = va_arg (*args, gpointer)))
    return FALSE;

  if (memcmp (param, "[", 2) == 0)
    return foundry_json_node_parse_array (node, args);

  if (memcmp (param, "{", 2) == 0)
    return foundry_json_node_parse_object (node, args);

  return FALSE;
}

gboolean
_foundry_json_node_parse (JsonNode *node,
                          ...)
{
  va_list args;
  gboolean ret;

  if (node == NULL)
    return FALSE;

  va_start (args, node);
  ret = foundry_json_node_parse_valist (node, &args);
  va_end (args);

  return ret;
}
