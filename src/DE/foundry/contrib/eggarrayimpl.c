/*
 * Copyright © 2020 Benjamin Otte
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library. If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Benjamin Otte <otte@gnome.org>
 */

#include <glib.h>

G_BEGIN_DECLS

#ifndef EGG_ARRAY_TYPE_NAME
#define EGG_ARRAY_TYPE_NAME EggArray
#endif

#ifndef EGG_ARRAY_NAME
#define EGG_ARRAY_NAME egg_array
#endif

#ifndef EGG_ARRAY_ELEMENT_TYPE
#define EGG_ARRAY_ELEMENT_TYPE gpointer
#endif

#ifdef EGG_ARRAY_PREALLOC
#if EGG_ARRAY_PREALLOC == 0
#undef EGG_ARRAY_PREALLOC
#endif
#endif

#ifdef EGG_ARRAY_NULL_TERMINATED
#define EGG_ARRAY_REAL_SIZE(_size) ((_size) + 1)
#define EGG_ARRAY_MAX_SIZE (G_MAXSIZE / sizeof (_T_) - 1)
#else
#define EGG_ARRAY_REAL_SIZE(_size) (_size)
#define EGG_ARRAY_MAX_SIZE (G_MAXSIZE / sizeof (_T_))
#endif

/* make this readable */
#define _T_ EGG_ARRAY_ELEMENT_TYPE
#define EggArray EGG_ARRAY_TYPE_NAME
#define egg_array_paste_more(EGG_ARRAY_NAME, func_name) EGG_ARRAY_NAME ## _ ## func_name
#define egg_array_paste(EGG_ARRAY_NAME, func_name) egg_array_paste_more (EGG_ARRAY_NAME, func_name)
#define egg_array(func_name) egg_array_paste (EGG_ARRAY_NAME, func_name)

typedef struct EggArray EggArray;

struct EggArray
{
  _T_ *start;
  _T_ *end;
  _T_ *end_allocation;
#ifdef EGG_ARRAY_PREALLOC
  _T_ preallocated[EGG_ARRAY_REAL_SIZE(EGG_ARRAY_PREALLOC)];
#endif
};

/* no G_GNUC_UNUSED here, if you don't use an array type, remove it. */
static inline void
egg_array(init) (EggArray *self)
{
#ifdef EGG_ARRAY_PREALLOC
  self->start = self->preallocated;
  self->end = self->start;
  self->end_allocation = self->start + EGG_ARRAY_PREALLOC;
#ifdef EGG_ARRAY_NULL_TERMINATED
  *self->start = *(_T_[1]) { 0 };
#endif
#else
  self->start = NULL;
  self->end = NULL;
  self->end_allocation = NULL;
#endif
}

G_GNUC_UNUSED static inline gsize
egg_array(get_capacity) (const EggArray *self)
{
  return self->end_allocation - self->start;
}

G_GNUC_UNUSED static inline gsize
egg_array(get_size) (const EggArray *self)
{
  return self->end - self->start;
}

static inline void
egg_array(free_elements) (_T_ *start,
                          _T_ *end)
{
#ifdef EGG_ARRAY_FREE_FUNC
  _T_ *e;
  for (e = start; e < end; e++)
#ifdef EGG_ARRAY_BY_VALUE
    EGG_ARRAY_FREE_FUNC (e);
#else
    EGG_ARRAY_FREE_FUNC (*e);
#endif
#endif
}

/* no G_GNUC_UNUSED here */
static inline void
egg_array(clear) (EggArray *self)
{
  egg_array(free_elements) (self->start, self->end);

#ifdef EGG_ARRAY_PREALLOC
  if (self->start != self->preallocated)
#endif
    g_free (self->start);
  egg_array(init) (self);
}

/*
 * egg_array_steal:
 * @self: the array
 *
 * Steals all data in the array and clears the array.
 *
 * If you need to know the size of the data, you should query it
 * beforehand.
 *
 * Returns: The array's data
 **/
G_GNUC_UNUSED static inline _T_ *
egg_array(steal) (EggArray *self)
{
  _T_ *result;

#ifdef EGG_ARRAY_PREALLOC
  if (self->start == self->preallocated)
    {
      gsize size = EGG_ARRAY_REAL_SIZE (egg_array(get_size) (self));
      result = g_new (_T_, size);
      memcpy (result, self->preallocated, sizeof (_T_) * size);
    }
  else
#endif
    result = self->start;

  egg_array(init) (self);

  return result;
}

G_GNUC_UNUSED static inline _T_ *
egg_array(get_data) (const EggArray *self)
{
  return self->start;
}

G_GNUC_UNUSED static inline _T_ *
egg_array(index) (const EggArray *self,
                  gsize           pos)
{
  return self->start + pos;
}

G_GNUC_UNUSED static inline gboolean
egg_array(is_empty) (const EggArray *self)
{
  return self->end == self->start;
}

G_GNUC_UNUSED static inline void
egg_array(reserve) (EggArray *self,
                    gsize      n)
{
  gsize new_capacity, size, capacity;

  if (G_UNLIKELY (n > EGG_ARRAY_MAX_SIZE))
    g_error ("requesting array size of %zu, but maximum size is %zu", n, EGG_ARRAY_MAX_SIZE);

  capacity = egg_array(get_capacity) (self);
  if (n <= capacity)
     return;

  size = egg_array(get_size) (self);
  /* capacity * 2 can overflow, that's why we MAX() */
  new_capacity = MAX (EGG_ARRAY_REAL_SIZE (n), capacity * 2);

#ifdef EGG_ARRAY_PREALLOC
  if (self->start == self->preallocated)
    {
      self->start = g_new (_T_, new_capacity);
      memcpy (self->start, self->preallocated, sizeof (_T_) * EGG_ARRAY_REAL_SIZE (size));
    }
  else
#endif
#ifdef EGG_ARRAY_NULL_TERMINATED
  if (self->start == NULL)
    {
      self->start = g_new (_T_, new_capacity);
      *self->start = *(_T_[1]) { 0 };
    }
  else
#endif
    self->start = g_renew (_T_, self->start, new_capacity);

  self->end = self->start + size;
  self->end_allocation = self->start + new_capacity;
#ifdef EGG_ARRAY_NULL_TERMINATED
  self->end_allocation--;
#endif
}

G_GNUC_UNUSED static inline void
egg_array(splice) (EggArray *self,
                   gsize      pos,
                   gsize      removed,
                   gboolean   stolen,
#ifdef EGG_ARRAY_BY_VALUE
                   const _T_ *additions,
#else
                   _T_       *additions,
#endif
                   gsize      added)
{
  gsize size;
  gsize remaining;

  size = egg_array(get_size) (self);
  g_assert (pos + removed <= size);
  remaining = size - pos - removed;

  if (!stolen)
    egg_array(free_elements) (egg_array(index) (self, pos),
                              egg_array(index) (self, pos + removed));

  egg_array(reserve) (self, size - removed + added);

  if (EGG_ARRAY_REAL_SIZE (remaining) && removed != added)
    memmove (egg_array(index) (self, pos + added),
             egg_array(index) (self, pos + removed),
             EGG_ARRAY_REAL_SIZE (remaining) * sizeof (_T_));

  if (added)
    {
      if (additions)
        memcpy (egg_array(index) (self, pos),
                additions,
                added * sizeof (_T_));
#ifndef EGG_ARRAY_NO_MEMSET
      else
        memset (egg_array(index) (self, pos), 0, added * sizeof (_T_));
#endif
    }


  /* might overflow, but does the right thing */
  self->end += added - removed;
}

G_GNUC_UNUSED static void
egg_array(set_size) (EggArray *self,
                     gsize     new_size)
{
  gsize old_size = egg_array(get_size) (self);
  if (new_size > old_size)
    egg_array(splice) (self, old_size, 0, FALSE, NULL, new_size - old_size);
  else
    egg_array(splice) (self, new_size, old_size - new_size, FALSE, NULL, 0);
}

G_GNUC_UNUSED static void
egg_array(append) (EggArray *self,
#ifdef EGG_ARRAY_BY_VALUE
                   _T_       *value)
#else
                   _T_        value)
#endif
{
  egg_array(splice) (self,
                     egg_array(get_size) (self),
                     0,
                     FALSE,
#ifdef EGG_ARRAY_BY_VALUE
                     value,
#else
                     &value,
#endif
                     1);
}

#ifdef EGG_ARRAY_BY_VALUE
G_GNUC_UNUSED static _T_ *
egg_array(get) (const EggArray *self,
                gsize           pos)
{
  return egg_array(index) (self, pos);
}
#else
G_GNUC_UNUSED static _T_
egg_array(get) (const EggArray *self,
                gsize           pos)
 {
   return *egg_array(index) (self, pos);
 }
#endif

#ifndef EGG_ARRAY_NO_UNDEF

#undef _T_
#undef EggArray
#undef egg_array_paste_more
#undef egg_array_paste
#undef egg_array
#undef EGG_ARRAY_REAL_SIZE
#undef EGG_ARRAY_MAX_SIZE

#undef EGG_ARRAY_BY_VALUE
#undef EGG_ARRAY_ELEMENT_TYPE
#undef EGG_ARRAY_FREE_FUNC
#undef EGG_ARRAY_NAME
#undef EGG_ARRAY_NULL_TERMINATED
#undef EGG_ARRAY_PREALLOC
#undef EGG_ARRAY_TYPE_NAME
#undef EGG_ARRAY_NO_MEMSET
#endif

G_END_DECLS
