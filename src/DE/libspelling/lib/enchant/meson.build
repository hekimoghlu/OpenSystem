libenchant_dep = dependency('enchant-2')
libicu_dep = dependency('icu-uc')

libspelling_deps += [libenchant_dep, libicu_dep]

libspelling_private_sources += files([
  'spelling-enchant-dictionary.c',
  'spelling-enchant-provider.c',
])
