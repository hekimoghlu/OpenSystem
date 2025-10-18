scalable_dir = 'hicolor' / 'scalable' / 'apps'
install_data(
  scalable_dir / application_id + '.svg',
  install_dir: decibels_datadir / 'icons' / scalable_dir
)

symbolic_dir = 'hicolor' / 'symbolic' / 'apps'
install_data(
  symbolic_dir / 'org.gnome.Decibels-symbolic.svg',
  rename: application_id + '-symbolic.svg',
  install_dir: decibels_datadir / 'icons' / symbolic_dir
)
