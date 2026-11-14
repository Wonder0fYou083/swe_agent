# slug

`slugify(title)` in `slug.py` turns a title into a URL slug:

- Letters become lowercase.
- ASCII letters and digits are kept.
- Every run of other characters becomes a single `-`.
- The slug never starts or ends with `-`.

```python
slugify("Hello, World!")     # "hello-world"
slugify("  Python 3.12  ")   # "python-3-12"
slugify("---")               # ""
```
