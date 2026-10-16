# Teacher prompt

`teacher_system.txt` is the final production prompt supplied with the original CoT generation code. It requests task understanding, scene analysis, and selection reasoning in Chinese. The downstream sanitizer supplies object coordinates, labels, and counts from the annotations. `generate_teacher.py` resolves this prompt relative to its own file, independently of the working directory.
