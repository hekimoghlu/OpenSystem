/* foundry-llm-tool.c
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

#include "foundry-llm-tool.h"
#include "foundry-util.h"

G_DEFINE_ABSTRACT_TYPE (FoundryLlmTool, foundry_llm_tool, FOUNDRY_TYPE_CONTEXTUAL)

enum {
  PROP_0,
  PROP_DESCRIPTION,
  PROP_NAME,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static GParamSpec **
foundry_llm_tool_real_list_parameters (FoundryLlmTool *self,
                                       guint          *n_parameters)
{
  GParamSpec **pspecs;

  g_assert (FOUNDRY_IS_LLM_TOOL (self));
  g_assert (n_parameters != NULL);

  *n_parameters = 0;

  if (!(pspecs = FOUNDRY_LLM_TOOL_GET_CLASS (self)->_parameters))
    return NULL;

  for (; pspecs[*n_parameters]; (*n_parameters)++) { }

  return g_memdup2 (pspecs, sizeof (GParamSpec *) * (*n_parameters));
}

static void
foundry_llm_tool_get_property (GObject    *object,
                               guint       prop_id,
                               GValue     *value,
                               GParamSpec *pspec)
{
  FoundryLlmTool *self = FOUNDRY_LLM_TOOL (object);

  switch (prop_id)
    {
    case PROP_DESCRIPTION:
      g_value_take_string (value, foundry_llm_tool_dup_description (self));
      break;

    case PROP_NAME:
      g_value_take_string (value, foundry_llm_tool_dup_name (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_llm_tool_class_init (FoundryLlmToolClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_llm_tool_get_property;

  klass->list_parameters = foundry_llm_tool_real_list_parameters;

  properties[PROP_DESCRIPTION] =
    g_param_spec_string ("description", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_llm_tool_init (FoundryLlmTool *self)
{
}

/**
 * foundry_llm_tool_dup_description:
 * @self: a [class@Foundry.LlmTool]
 *
 * Gets the "description" of a tool which may be provided to the model.
 *
 * This should describe what the function does.
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_llm_tool_dup_description (FoundryLlmTool *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_TOOL (self), NULL);

  if (FOUNDRY_LLM_TOOL_GET_CLASS (self)->dup_description)
    return FOUNDRY_LLM_TOOL_GET_CLASS (self)->dup_description (self);

  return NULL;
}

/**
 * foundry_llm_tool_dup_name:
 * @self: a [class@Foundry.LlmTool]
 *
 * Gets the "name" of tool which may be provided to the model.
 *
 * This is often a descriptive function name like "getWeather".
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_llm_tool_dup_name (FoundryLlmTool *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_TOOL (self), NULL);

  if (FOUNDRY_LLM_TOOL_GET_CLASS (self)->dup_name)
    return FOUNDRY_LLM_TOOL_GET_CLASS (self)->dup_name (self);

  return NULL;
}

/**
 * foundry_llm_tool_list_parameters:
 * @self: a [class@Foundry.LlmTool]
 * @n_parameters: (out):
 *
 * Gets a list of parmaeters for the tool which must be
 * supplied when calling [method@Foundry.LlmTool.call].
 *
 * Returns: (transfer container) (array length=n_parameters) (nullable):
 */
GParamSpec **
foundry_llm_tool_list_parameters (FoundryLlmTool *self,
                                  guint          *n_parameters)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_TOOL (self), NULL);
  g_return_val_if_fail (n_parameters != NULL, NULL);

  *n_parameters = 0;

  if (FOUNDRY_LLM_TOOL_GET_CLASS (self)->list_parameters)
    return FOUNDRY_LLM_TOOL_GET_CLASS (self)->list_parameters (self, n_parameters);

  return NULL;
}

/**
 * foundry_llm_tool_call:
 * @self: a [class@Foundry.LlmTool]
 * @arguments: (array length=n_arguments):
 * @n_arguments: number of values in @arguments
 *
 * Calls the tool.
 *
 * @arguments should be an array of initialized `GValue` which contain
 * values set matching each of the parameters specified by
 * [method@Foundry.LlmTool.list_parameters] in the same order.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.LlmMessage] or rejects with error.
 */
DexFuture *
foundry_llm_tool_call (FoundryLlmTool *self,
                       const GValue   *arguments,
                       guint           n_arguments)
{
  g_return_val_if_fail (FOUNDRY_IS_LLM_TOOL (self), NULL);
  g_return_val_if_fail (arguments != NULL || n_arguments == 0, NULL);

  for (guint i = 0; i < n_arguments; i++)
    g_return_val_if_fail (G_IS_VALUE (&arguments[i]), NULL);

  if (FOUNDRY_LLM_TOOL_GET_CLASS (self)->call)
    return FOUNDRY_LLM_TOOL_GET_CLASS (self)->call (self, arguments, n_arguments);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_llm_tool_class_add_parameter:
 * @pspec: (transfer full): the parameter to add
 */
void
foundry_llm_tool_class_add_parameter (FoundryLlmToolClass *tool_class,
                                      GParamSpec          *pspec)
{
  g_return_if_fail (FOUNDRY_IS_LLM_TOOL_CLASS (tool_class));
  g_return_if_fail (G_IS_PARAM_SPEC (pspec));

  if (tool_class->_parameters == NULL)
    {
      GParamSpec **pspecs;

      pspecs = tool_class->_parameters = g_new0 (GParamSpec *, 2);
      pspecs[0] = pspec;
    }
  else
    {
      GParamSpec **pspecs = tool_class->_parameters;
      gsize size;

      for (size = 0; pspecs[size]; size++) { }

      tool_class->_parameters = g_realloc_n (tool_class->_parameters, size + 2, sizeof (GParamSpec *));
      pspecs = tool_class->_parameters;

      pspecs[size++] = pspec;
      pspecs[size] = NULL;
    }
}
