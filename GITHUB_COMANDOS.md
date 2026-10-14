# Comandos para trabajar con el repositorio de GitHub

Repositorio: <https://github.com/wammcode/transcription-a2s-model_amt>

La rama principal es `main` y el remoto de GitHub se llama `origin`.
Los comandos siguientes se ejecutan en una terminal con Git instalado.

## 1. Clonar el proyecto por primera vez

Desde la carpeta donde quieras guardar el proyecto:

```bash
git clone https://github.com/wammcode/transcription-a2s-model_amt.git
cd transcription-a2s-model_amt
```

El proyecto completo se gestiona desde su carpeta raíz. `Transkun/`,
`a2s-transformer/` y las carpetas de los papers forman parte del mismo
repositorio; no requieren clonar submódulos.

## 2. Consultar el estado

Desde la carpeta raíz del proyecto:

```bash
git status
git remote -v
git branch -vv
git log -5 --oneline
```

## 3. Actualizar la copia local desde GitHub

Con los cambios locales ya confirmados o guardados, ejecuta:

```bash
git switch main
git pull --ff-only origin main
```

`--ff-only` actualiza la rama si puede hacerlo sin crear un commit de mezcla.
Si Git indica que las ramas han divergido, sigue el apartado 6.

Si todavía tienes cambios locales sin confirmar y quieres guardarlos
temporalmente antes de actualizar:

```bash
git stash push -u -m "Cambios locales antes de actualizar"
git switch main
git pull --ff-only origin main
git stash pop
```

Este ejemplo supone que tus cambios pertenecen a `main`. `-u` incluye archivos
nuevos sin seguimiento, pero respeta los archivos ignorados. Si aparecen
conflictos al recuperar los cambios, resuélvelos antes de continuar.

## 4. Subir cambios locales a GitHub

Trabajando en `main`, revisa, prepara y confirma los cambios:

```bash
git status
git diff
git add -A
git diff --cached
git commit -m "Describe los cambios realizados"
git push origin main
```

`git add -A` prepara archivos nuevos, modificaciones y eliminaciones dentro
del repositorio. Revisa `git diff --cached` antes de confirmar. Para subir solo
algunos archivos, sustituye `git add -A` por `git add ruta/al/archivo`.

Los archivos excluidos mediante `.gitignore`, como entornos virtuales, cachés
y `Transkun/artifacts/`, se conservan localmente y no se añaden con este flujo.
El modelo preentrenado `Transkun/transkun/pretrained/2.0.pt` ya forma parte
del historial y se incluye al clonar.

## 5. Comprobar que los cambios están sincronizados

```bash
git fetch origin
git status
git rev-parse HEAD
git rev-parse origin/main
```

Si estás en `main`, los dos hashes deben coincidir después de subir todos
los commits. `git status` también debe indicar que no hay cambios pendientes.

## 6. Si GitHub tiene commits que aún no tienes localmente

Si `git push` se rechaza porque la rama remota ha avanzado, confirma primero
tus cambios y, con el directorio de trabajo limpio, ejecuta:

```bash
git switch main
git pull --rebase origin main
git push origin main
```

Si aparecen conflictos, corrige los archivos indicados, añade cada archivo
resuelto con `git add ruta/al/archivo` y ejecuta `git rebase --continue`.
Repite si hace falta y después vuelve a ejecutar `git push origin main`.
Para cancelar el rebase y regresar al estado anterior: `git rebase --abort`.

## 7. Autenticación y configuración del remoto

Para subir por HTTPS necesitas autenticación de GitHub y permiso de escritura
en este repositorio. Si Git solicita usuario y contraseña, utiliza tu usuario
de GitHub y un token de acceso personal con permisos para este repositorio
como contraseña. También puedes usar un gestor de credenciales configurado.

Si `origin` no existe en una copia local que ya tiene historial:

```bash
git remote add origin https://github.com/wammcode/transcription-a2s-model_amt.git
git push -u origin main
```

Si `origin` existe pero apunta a otra dirección:

```bash
git remote set-url origin https://github.com/wammcode/transcription-a2s-model_amt.git
```

Si tienes una clave SSH registrada en GitHub, puedes utilizar SSH:

```bash
git remote set-url origin git@github.com:wammcode/transcription-a2s-model_amt.git
git push -u origin main
```

Para configurar la identidad de los próximos commits solo en este proyecto,
sustituye los valores de ejemplo:

```bash
git config user.name "Tu nombre"
git config user.email "tu-correo@example.com"
```
